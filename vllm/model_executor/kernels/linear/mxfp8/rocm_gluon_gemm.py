# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Portions adapted from ROCm/gfx950-gluon-tutorials
# (https://github.com/ROCm/gfx950-gluon-tutorials, kernels/gemm/inter_wave/a8w8):
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AMD ROCm™ Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Gluon MXFP8 GEMM for gfx950 on a (16, 16)-preshuffled weight, large M.

`y = x @ w.T` for the inputs AITER's FlyDSL GEMM takes: an e4m3 activation
with row-major E8M0 scales `[M, K / 32]` and an e4m3 weight in AITER's
`(16, 16)` preshuffled layout. The weight scale is read expanded per row and
transposed, `[K / 32, N]` (`weight_scale_cols`).

A workgroup is 8 warps on a 256 x 256 tile cut into 2 x 2 quadrants, with
four double-buffered LDS operand buffers and a warp pipeline that keeps one
MFMA stream per SIMD, as in ROCm/gfx950-gluon-tutorials
`kernels/gemm/inter_wave/a8w8`. What the rest of the time depends on:

* scales sit in LDS with the rows of one K group adjacent, so an MFMA scale
  operand is one transposed 64-bit LDS read. The activation scale goes to an
  LDS staging buffer as it is and is moved there a K-pair later;
* the result is stored from a non-transposed MFMA layout (a lane holds
  consecutive rows), so a store touches few rows: 4 with 16x16x128, or 2
  whole cache lines with 32x32x64, which wins when K is short;
* tiles are ordered so that the larger operand stays in an XCD's L2;
* when the tiles do not divide evenly among the CUs, the last columns run as
  256 x 128 tiles, a full last round of half-size work.

Only shapes in `_TUNED` run here (measured against AITER v0.1.29 on MI350X);
`gluon_gemm_enabled` says for which M.
"""

from functools import cache
from typing import NamedTuple

import torch

from vllm.triton_utils import aggregate, gl, gluon, tl
from vllm.utils.platform_utils import num_compute_units

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
_NUM_WARPS = 8
_NUM_CUS = 256
# A half tile takes this much of a full tile's time.
_HALF_TILE_COST = 0.62
# Copies into LDS are 16 bytes wide, which Triton has to prove of the pointers.
_ALIGN = 16


class GluonGemmConfig(NamedTuple):
    """Tuned launch parameters of one `(N, K)`."""

    mfma32: bool  # 32x32x64 MFMA instead of 16x16x128
    group_m: int  # activation tiles an XCD keeps in L2 (when group_n == 0)
    group_n: int  # weight tiles an XCD keeps in L2 instead, or 0
    min_row_tiles: int  # AITER's GEMM is faster for fewer rows of tiles,
    row_tiles: tuple[int, ...] = ()  # except for these counts


# (N, K) -> config, DeepSeek V4.1 dense attention linears at TP=1, tuned for
# the tile rounds of `_NUM_CUS` CUs with
# benchmarks/kernels/benchmark_rocm_mxfp8_gluon_gemm.py (cold weights).
_TUNED: dict[tuple[int, int], GluonGemmConfig] = {
    (32768, 1280): GluonGemmConfig(True, 8, 6, 6, (4,)),  # wq_b
    (5120, 8192): GluonGemmConfig(False, 8, 0, 13),  # wo_b
}


@aggregate
class _Layouts:
    """Layouts of a tile, by MFMA shape. Tensors are [M, K], [K, N] or
    [rows, K groups]; LDS paddings and base orders are tuned per shape."""

    mfma: gl.constexpr
    dot_a: gl.constexpr
    dot_b: gl.constexpr
    load_a: gl.constexpr
    load_b: gl.constexpr
    shared_a: gl.constexpr
    shared_b: gl.constexpr
    scale_a: gl.constexpr
    scale_b: gl.constexpr
    scale_a_pair: gl.constexpr
    scale_b_pair: gl.constexpr
    scale_fill: gl.constexpr
    scale_rows: gl.constexpr
    scale_quad: gl.constexpr
    scale_shared: gl.constexpr
    scale_stage: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, MFMA32):
        hm, hn, ng = BLOCK_M // 2, BLOCK_N // 2, BLOCK_K // 32
        mfma = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 64] if MFMA32 else [16, 16, 128],
            transposed=False,
            warps_per_cta=[2, 4],
        )
        dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
        dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
        k_bytes = [[0, 1], [0, 2], [0, 4], [0, 8]]
        k_chunks = [[0, 16], [0, 32], [0, 64]]
        if MFMA32:
            rows = [[1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0]]
            pad = [[1024, 16]]
        else:
            rows = [[16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
            pad = [[1024, 32]]
        # A half tile: 16 K bytes per lane, 8 rows per copy.
        load_a = gl.DistributedLinearLayout(
            reg_bases=k_bytes + rows[6:],
            lane_bases=k_chunks + rows[:3],
            warp_bases=rows[3:6],
            block_bases=[],
            shape=[hm, BLOCK_K],
        )
        shared_a = gl.PaddedSharedLayout(
            pad, k_bytes + k_chunks + rows, [], [hm, BLOCK_K]
        )
        # B half tile from the preshuffled weight: a copy reads 1 KB of
        # consecutive memory (16 rows x 4 K chunks), a warp one 16-row block.
        b_bytes = [[1, 0], [2, 0], [4, 0], [8, 0]]
        b_lanes = [[0, 1], [0, 2], [0, 4], [0, 8], [16, 0], [32, 0]]
        b_warps = [[0, 16], [0, 32], [0, 64]]
        load_b = gl.DistributedLinearLayout(
            reg_bases=b_bytes + [[64, 0]],
            lane_bases=b_lanes,
            warp_bases=b_warps,
            block_bases=[],
            shape=[BLOCK_K, hn],
        )
        shared_b = gl.PaddedSharedLayout(
            pad, b_bytes + b_lanes + b_warps + [[64, 0]], [], [BLOCK_K, hn]
        )
        scale_layout = gl.amd.cdna4.get_mfma_scale_layout
        scale_a = scale_layout(dot_a, [hm, ng])
        scale_b = scale_layout(dot_b, [hn, ng])
        if MFMA32:
            scale_a_pair = scale_layout(dot_a, [BLOCK_M, 2 * ng])
            scale_b_pair = scale_layout(dot_b, [BLOCK_N, 2 * ng])
        else:
            # The K-step and the tile half as the high register bits: a
            # K-pair splits into its operands by register slices.
            lanes = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
            scale_a_pair = gl.DistributedLinearLayout(
                reg_bases=[[32, 0], [64, 0], [0, 4], [128, 0]],
                lane_bases=lanes,
                warp_bases=[[0, 0], [0, 0], [16, 0]],
                block_bases=[],
                shape=[BLOCK_M, 2 * ng],
            )
            scale_b_pair = gl.DistributedLinearLayout(
                reg_bases=[[64, 0], [128, 0], [0, 4]],
                lane_bases=lanes,
                warp_bases=[[16, 0], [32, 0], [0, 0]],
                block_bases=[],
                shape=[BLOCK_N, 2 * ng],
            )
        self.mfma = gl.constexpr(mfma)
        self.dot_a = gl.constexpr(dot_a)
        self.dot_b = gl.constexpr(dot_b)
        self.load_a = gl.constexpr(load_a)
        self.load_b = gl.constexpr(load_b)
        self.shared_a = gl.constexpr(shared_a)
        self.shared_b = gl.constexpr(shared_b)
        self.scale_a = gl.constexpr(scale_a)
        self.scale_b = gl.constexpr(scale_b)
        self.scale_a_pair = gl.constexpr(scale_a_pair)
        self.scale_b_pair = gl.constexpr(scale_b_pair)
        # [K / 32, rows] scale -> LDS: a warp copies one K group x 256 rows.
        self.scale_fill = gl.constexpr(
            gl.BlockedLayout([4, 1], [64, 1], [1, 8], [0, 1])
        )
        # [M, K / 32] scale -> LDS staging: a warp copies 32 rows x 8 bytes.
        self.scale_rows = gl.constexpr(
            gl.BlockedLayout([1, 4], [32, 2], [8, 1], [1, 0])
        )
        # Staging -> scale buffer: a lane moves 4 rows of one K group, so the
        # (slow) LDS store is one dword store per warp.
        self.scale_quad = gl.constexpr(gl.BlockedLayout([4, 1], [8, 8], [8, 1], [0, 1]))
        self.scale_shared = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1]))
        self.scale_stage = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]))


@gluon.jit
def _tile_pids(
    pid,
    NM,
    NN,
    GROUP_M: gl.constexpr,
    GROUP_N: gl.constexpr,
):
    """Tile of workgroup `pid` in an NM x NN grid: XCDs take programs round
    robin, so each gets a run of tiles that share GROUP_M activation tiles
    (or, with GROUP_N, GROUP_N weight tiles)."""
    XCDS: gl.constexpr = 8
    grid = NM * NN
    per_xcd = (grid + XCDS - 1) // XCDS
    tall = (grid - 1) % XCDS + 1
    xcd = pid % XCDS
    local = pid // XCDS
    if xcd < tall:
        pid = xcd * per_xcd + local
    else:
        pid = tall * per_xcd + (xcd - tall) * (per_xcd - 1) + local
    if GROUP_N > 0:
        in_group = GROUP_N * NM
        first_n = (pid // in_group) * GROUP_N
        size_n = min(NN - first_n, GROUP_N)
        pid_n = first_n + (pid % in_group) % size_n
        pid_m = (pid % in_group) // size_n
    else:
        in_group = GROUP_M * NN
        first_m = (pid // in_group) * GROUP_M
        size_m = min(NM - first_m, GROUP_M)
        pid_m = first_m + (pid % in_group) % size_m
        pid_n = (pid % in_group) // size_m
    return pid_m, pid_n


@gluon.jit
def _split_halves(s, half_layout: gl.constexpr):
    """[2 * R, 8] K-pair scale -> the K-step 0 and 1 operands of each half
    of its rows: (first 0, first 1, second 0, second 1)."""
    R: gl.constexpr = s.shape[0] // 2
    first, second = gl.split(gl.permute(s.reshape([2, R, 2, 4]), [1, 3, 2, 0]))
    f0, f1 = gl.split(first)
    s0, s1 = gl.split(second)
    return (
        gl.convert_layout(f0, half_layout),
        gl.convert_layout(f1, half_layout),
        gl.convert_layout(s0, half_layout),
        gl.convert_layout(s1, half_layout),
    )


@gluon.jit
def _mxfp8_gluon_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    K,
    stride_cm,
    a_scale_stride4,
    b_scale_stride4,
    m0,
    n0,
    lay: _Layouts,
    HALF: gl.constexpr,
):
    """Rows `m0..m0 + 256` (`m0` is int64: M * N can pass 2^31) of a
    256-column window at column `n0`: all of it, or with HALF only its right
    128 columns (a half tile; its window starts 128 columns left of it,
    inside the matrix as full tiles come first).

    A K-pair (256 K) is 8 MFMA stages on the quadrants tl, bl, tr, br (4 on
    tr, br for a half tile), each followed by a memory stage that reads the
    next operand half from LDS and issues copies. Pair q also reads pair
    q + 1's scales, copied a pair earlier, and copies those of pair q + 2.
    A copy is read 6 stages after it was issued (2 in a half tile).
    """
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    BLOCK_K: gl.constexpr = 128
    HM: gl.constexpr = BLOCK_M // 2
    HN: gl.constexpr = BLOCK_N // 2
    NG: gl.constexpr = BLOCK_K // 32
    WAIT: gl.constexpr = 2 if HALF else 5

    f8 = a_ptr.dtype.element_ty
    u8 = a_scale_ptr.dtype.element_ty
    smem_a_top = gl.allocate_shared_memory(f8, [2, HM, BLOCK_K], lay.shared_a)
    smem_a_bot = gl.allocate_shared_memory(f8, [2, HM, BLOCK_K], lay.shared_a)
    smem_b_right = gl.allocate_shared_memory(f8, [2, BLOCK_K, HN], lay.shared_b)
    smem_b_left = smem_b_right
    if not HALF:
        smem_b_left = gl.allocate_shared_memory(f8, [2, BLOCK_K, HN], lay.shared_b)
    smem_stage = gl.allocate_shared_memory(u8, [2, BLOCK_M, 2 * NG], lay.scale_stage)
    smem_sa = gl.allocate_shared_memory(u8, [2, BLOCK_M, 2 * NG], lay.scale_shared)
    smem_sb = gl.allocate_shared_memory(u8, [2, BLOCK_N, 2 * NG], lay.scale_shared)

    # K is contiguous in A; the weight is [N / 16, K / 32, 2, 16, 16].
    offs_am = gl.arange(0, HM, gl.SliceLayout(1, lay.load_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, lay.load_a))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, lay.load_b))
    offs_bn = gl.arange(0, HN, gl.SliceLayout(0, lay.load_b))
    a_offs = offs_am[:, None] * K + offs_ak[None, :]
    b_offs = (
        (offs_bk // 16)[:, None] * 256
        + (offs_bk % 16)[:, None]
        + (offs_bn % 16)[None, :] * 16
        + (offs_bn // 16)[None, :] * (16 * K)
    )
    a_base = a_ptr + m0 * K
    b_base = b_ptr + n0 * K
    A_HALF = HM * K
    B_HALF = HN * K
    B_KSTEP: gl.constexpr = 16 * BLOCK_K

    offs_skb = gl.arange(0, 2 * NG, gl.SliceLayout(0, lay.scale_fill))
    offs_sn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, lay.scale_fill))
    SB_K = b_scale_stride4 * 4
    sb_offs = offs_sn[:, None] + offs_skb[None, :] * SB_K
    sb_tile = b_scale_ptr + n0
    offs_ska = gl.arange(0, 2 * NG, gl.SliceLayout(0, lay.scale_rows))
    offs_sm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, lay.scale_rows))
    SA_ROW = a_scale_stride4 * 4
    sa_offs = offs_sm[:, None] * SA_ROW + offs_ska[None, :]
    sa_tile = a_scale_ptr + m0 * SA_ROW

    acc_tl = gl.zeros((HM, HN), gl.float32, lay.mfma)
    acc_bl = gl.zeros((HM, HN), gl.float32, lay.mfma)
    acc_tr = gl.zeros((HM, HN), gl.float32, lay.mfma)
    acc_br = gl.zeros((HM, HN), gl.float32, lay.mfma)

    k_steps = K // BLOCK_K
    last = k_steps // 2 - 1

    # K-steps 0 and 1 into buffers 0 and 1, the scales of K-pairs 0 and 1.
    if not HALF:
        async_copy.buffer_load_to_shared(smem_b_left.index(0), b_base, b_offs)
    else:
        async_copy.buffer_load_to_shared(smem_b_right.index(0), b_base + B_HALF, b_offs)
        async_copy.buffer_load_to_shared(smem_a_top.index(0), a_base, a_offs)
    async_copy.buffer_load_to_shared(smem_stage.index(0), sa_tile, sa_offs)
    async_copy.buffer_load_to_shared(smem_stage.index(1), sa_tile + 2 * NG, sa_offs)
    async_copy.buffer_load_to_shared(smem_sb.index(0), sb_tile, sb_offs)
    async_copy.buffer_load_to_shared(smem_sb.index(1), sb_tile + 2 * NG * SB_K, sb_offs)
    async_copy.commit_group()
    if not HALF:
        async_copy.buffer_load_to_shared(smem_a_top.index(0), a_base, a_offs)
        async_copy.commit_group()
    async_copy.buffer_load_to_shared(smem_a_bot.index(0), a_base + A_HALF, a_offs)
    async_copy.commit_group()
    if not HALF:
        async_copy.buffer_load_to_shared(smem_b_right.index(0), b_base + B_HALF, b_offs)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smem_b_left.index(1), b_base + B_KSTEP, b_offs)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(smem_a_top.index(1), a_base + BLOCK_K, a_offs)
        async_copy.commit_group()
    else:
        async_copy.buffer_load_to_shared(
            smem_b_right.index(1), b_base + B_HALF + B_KSTEP, b_offs
        )
        async_copy.buffer_load_to_shared(smem_a_top.index(1), a_base + BLOCK_K, a_offs)
        async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        smem_a_bot.index(1), a_base + A_HALF + BLOCK_K, a_offs
    )
    async_copy.commit_group()
    if not HALF:
        async_copy.buffer_load_to_shared(
            smem_b_right.index(1), b_base + B_HALF + B_KSTEP, b_offs
        )
        async_copy.commit_group()
    a_base += 2 * BLOCK_K
    b_base += 2 * B_KSTEP

    async_copy.wait_group(3 if HALF else 6)
    # The operand a tile does not start with is a placeholder.
    b_left = smem_b_right.index(0).load(lay.dot_b)
    if not HALF:
        b_left = smem_b_left.index(0).load(lay.dot_b)
    b_right = b_left
    a_top = smem_a_top.index(0).load(lay.dot_a)
    a_bot = a_top
    rows = smem_stage.index(0).load(lay.scale_quad)
    smem_sa.index(0).store(rows)
    sa_top_0, sa_top_1, sa_bot_0, sa_bot_1 = _split_halves(
        smem_sa.index(0).load(lay.scale_a_pair), lay.scale_a
    )
    sb_left_0, sb_left_1, sb_right_0, sb_right_1 = _split_halves(
        smem_sb.index(0).load(lay.scale_b_pair), lay.scale_b
    )

    gl.assume(k_steps > 3)

    for k in tl.range(0, k_steps - 2, 2):
        q = k // 2
        nb = (q + 1) % 2
        sa_top = (sa_top_0, sa_top_1)
        sa_bot = (sa_bot_0, sa_bot_1)
        sb_left = (sb_left_0, sb_left_1)
        sb_right = (sb_right_0, sb_right_1)
        for j in gl.static_range(2):
            # Only waits may sit between stages: pointers are formed inside.
            if not HALF:
                async_copy.wait_group(WAIT)
                with gl.amd.warp_pipeline_stage("mfma", priority=0):
                    acc_tl = cdna4.mfma_scaled(
                        a_top, sa_top[j], "e4m3", b_left, sb_left[j], "e4m3", acc_tl
                    )
                with gl.amd.warp_pipeline_stage("mem", priority=1):
                    a_bot = smem_a_bot.index(j).load(lay.dot_a)
                    async_copy.buffer_load_to_shared(
                        smem_b_left.index(j), b_base + j * B_KSTEP, b_offs
                    )
                    async_copy.commit_group()
                async_copy.wait_group(WAIT)
                with gl.amd.warp_pipeline_stage("mfma", priority=0):
                    acc_bl = cdna4.mfma_scaled(
                        a_bot, sa_bot[j], "e4m3", b_left, sb_left[j], "e4m3", acc_bl
                    )
                with gl.amd.warp_pipeline_stage("mem", priority=1):
                    b_right = smem_b_right.index(j).load(lay.dot_b)
                    async_copy.buffer_load_to_shared(
                        smem_a_top.index(j), a_base + j * BLOCK_K, a_offs
                    )
                    if j == 0:
                        async_copy.buffer_load_to_shared(
                            smem_sb.index(q % 2),
                            sb_tile + 8 * min(q + 2, last) * SB_K,
                            sb_offs,
                        )
                        async_copy.buffer_load_to_shared(
                            smem_stage.index(q % 2),
                            sa_tile + 8 * min(q + 2, last),
                            sa_offs,
                        )
                    else:
                        smem_sa.index(nb).store(rows)
                    async_copy.commit_group()
            async_copy.wait_group(WAIT)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_tr = cdna4.mfma_scaled(
                    a_top, sa_top[j], "e4m3", b_right, sb_right[j], "e4m3", acc_tr
                )
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                if not HALF:
                    b_left = smem_b_left.index(1 - j).load(lay.dot_b)
                    async_copy.buffer_load_to_shared(
                        smem_a_bot.index(j), a_base + A_HALF + j * BLOCK_K, a_offs
                    )
                    if j == 0:
                        rows = smem_stage.index(nb).load(lay.scale_quad)
                    else:
                        sa_top_0, sa_top_1, sa_bot_0, sa_bot_1 = _split_halves(
                            smem_sa.index(nb).load(lay.scale_a_pair), lay.scale_a
                        )
                        sb_left_0, sb_left_1, sb_right_0, sb_right_1 = _split_halves(
                            smem_sb.index(nb).load(lay.scale_b_pair),
                            lay.scale_b,
                        )
                else:
                    a_bot = smem_a_bot.index(j).load(lay.dot_a)
                    async_copy.buffer_load_to_shared(
                        smem_b_right.index(j), b_base + B_HALF + j * B_KSTEP, b_offs
                    )
                    async_copy.buffer_load_to_shared(
                        smem_a_top.index(j), a_base + j * BLOCK_K, a_offs
                    )
                    if j == 0:
                        async_copy.buffer_load_to_shared(
                            smem_sb.index(q % 2),
                            sb_tile + 8 * min(q + 2, last) * SB_K,
                            sb_offs,
                        )
                        async_copy.buffer_load_to_shared(
                            smem_stage.index(q % 2),
                            sa_tile + 8 * min(q + 2, last),
                            sa_offs,
                        )
                    else:
                        smem_sa.index(nb).store(rows)
                async_copy.commit_group()
            async_copy.wait_group(WAIT)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc_br = cdna4.mfma_scaled(
                    a_bot, sa_bot[j], "e4m3", b_right, sb_right[j], "e4m3", acc_br
                )
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a_top = smem_a_top.index(1 - j).load(lay.dot_a)
                if not HALF:
                    async_copy.buffer_load_to_shared(
                        smem_b_right.index(j), b_base + B_HALF + j * B_KSTEP, b_offs
                    )
                else:
                    b_right = smem_b_right.index(1 - j).load(lay.dot_b)
                    async_copy.buffer_load_to_shared(
                        smem_a_bot.index(j), a_base + A_HALF + j * BLOCK_K, a_offs
                    )
                    if j == 0:
                        rows = smem_stage.index(nb).load(lay.scale_quad)
                    else:
                        sa_top_0, sa_top_1, sa_bot_0, sa_bot_1 = _split_halves(
                            smem_sa.index(nb).load(lay.scale_a_pair), lay.scale_a
                        )
                        sb_left_0, sb_left_1, sb_right_0, sb_right_1 = _split_halves(
                            smem_sb.index(nb).load(lay.scale_b_pair),
                            lay.scale_b,
                        )
                async_copy.commit_group()
        a_base += 2 * BLOCK_K
        b_base += 2 * B_KSTEP

    # The last K-pair has nothing left to copy; then the stores, straight
    # from the MFMA layout.
    offs_cm = gl.arange(0, HM, gl.SliceLayout(1, lay.mfma))
    offs_cn = gl.arange(0, HN, gl.SliceLayout(0, lay.mfma))
    c_offs = stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_tl = c_ptr + m0 * stride_cm + n0
    c_bl = c_tl + HM * stride_cm
    out_ty = c_ptr.dtype.element_ty
    sa_top = (sa_top_0, sa_top_1)
    sa_bot = (sa_bot_0, sa_bot_1)
    sb_left = (sb_left_0, sb_left_1)
    sb_right = (sb_right_0, sb_right_1)
    for j in gl.static_range(2):
        if not HALF:
            acc_tl = cdna4.mfma_scaled(
                a_top, sa_top[j], "e4m3", b_left, sb_left[j], "e4m3", acc_tl
            )
            async_copy.wait_group(5 - 4 * j)
            a_bot = smem_a_bot.index(j).load(lay.dot_a)
            acc_bl = cdna4.mfma_scaled(
                a_bot, sa_bot[j], "e4m3", b_left, sb_left[j], "e4m3", acc_bl
            )
            async_copy.wait_group(4 - 4 * j)
            b_right = smem_b_right.index(j).load(lay.dot_b)
            acc_tr = cdna4.mfma_scaled(
                a_top, sa_top[j], "e4m3", b_right, sb_right[j], "e4m3", acc_tr
            )
            if j == 0:
                async_copy.wait_group(3)
                b_left = smem_b_left.index(1).load(lay.dot_b)
            acc_br = cdna4.mfma_scaled(
                a_bot, sa_bot[j], "e4m3", b_right, sb_right[j], "e4m3", acc_br
            )
            if j == 0:
                async_copy.wait_group(2)
                a_top = smem_a_top.index(1).load(lay.dot_a)
        else:
            acc_tr = cdna4.mfma_scaled(
                a_top, sa_top[j], "e4m3", b_right, sb_right[j], "e4m3", acc_tr
            )
            async_copy.wait_group(2 - 2 * j)
            a_bot = smem_a_bot.index(j).load(lay.dot_a)
            acc_br = cdna4.mfma_scaled(
                a_bot, sa_bot[j], "e4m3", b_right, sb_right[j], "e4m3", acc_br
            )
            if j == 0:
                async_copy.wait_group(1)
                b_right = smem_b_right.index(1).load(lay.dot_b)
                a_top = smem_a_top.index(1).load(lay.dot_a)
    if not HALF:
        cdna4.buffer_store(ptr=c_tl, offsets=c_offs, stored_value=acc_tl.to(out_ty))
        cdna4.buffer_store(ptr=c_bl, offsets=c_offs, stored_value=acc_bl.to(out_ty))
    cdna4.buffer_store(ptr=c_tl + HN, offsets=c_offs, stored_value=acc_tr.to(out_ty))
    cdna4.buffer_store(ptr=c_bl + HN, offsets=c_offs, stored_value=acc_br.to(out_ty))


@gluon.jit(
    do_not_specialize=["M", "NM", "NN_FULL", "NN_HALF"],
    do_not_specialize_on_alignment=["c_ptr"],
)
def _mxfp8_gluon_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    K,
    stride_cm,
    a_scale_stride4,
    b_scale_stride4,
    NM,
    NN_FULL,
    NN_HALF,
    MFMA32: gl.constexpr,
    GROUP_M: gl.constexpr,
    GROUP_N: gl.constexpr,
):
    """Programs [0, NM * NN_FULL) take the full tiles of the first
    NN_FULL * 256 columns, the rest the half tiles of the NN_HALF * 128
    columns after them. The last row of tiles ends at row M, overlapping the
    one before it when M % 256 != 0 (both store the same values). Scale
    strides are / 4, so that copy offsets stay provably dword aligned."""
    BLOCK_M: gl.constexpr = 256
    BLOCK_N: gl.constexpr = 256
    lay = _Layouts(MFMA32)
    pid = gl.program_id(axis=0)
    if pid < NM * NN_FULL:
        pid_m, pid_n = _tile_pids(pid, NM, NN_FULL, GROUP_M, GROUP_N)
        _mxfp8_gluon_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            a_scale_ptr,
            b_scale_ptr,
            K,
            stride_cm,
            a_scale_stride4,
            b_scale_stride4,
            min(pid_m * BLOCK_M, M - BLOCK_M).to(gl.int64),
            pid_n * BLOCK_N,
            lay,
            False,
        )
    else:
        pid_m, pid_n = _tile_pids(pid - NM * NN_FULL, NM, NN_HALF, GROUP_M, GROUP_N)
        _mxfp8_gluon_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            a_scale_ptr,
            b_scale_ptr,
            K,
            stride_cm,
            a_scale_stride4,
            b_scale_stride4,
            min(pid_m * BLOCK_M, M - BLOCK_M).to(gl.int64),
            NN_FULL * BLOCK_N + (pid_n - 1) * (BLOCK_N // 2),
            lay,
            True,
        )


@cache
def _full_column_tiles(nm: int, nn: int) -> int:
    """How many of the nn 256-column tiles to run whole (the other columns as
    half tiles) for the fewest rounds of workgroups over the CUs."""

    def rounds(nf: int) -> float:
        half = -(-nm * 2 * (nn - nf) // _NUM_CUS)
        return -(-nm * nf // _NUM_CUS) + _HALF_TILE_COST * half

    return min(range(nn, 0, -1), key=rounds)


def gluon_gemm_config(N: int, K: int) -> GluonGemmConfig | None:
    """The tuned config of an `(N, K)` weight on this GPU, or None."""
    return _TUNED.get((N, K)) if num_compute_units() == _NUM_CUS else None


def gluon_gemm_enabled(M: int, N: int, K: int) -> bool:
    """Whether `rocm_mxfp8_gluon_gemm` is the faster GEMM for an `[M, K]`
    activation on an `[N, K]` weight (else AITER's)."""
    cfg = gluon_gemm_config(N, K)
    if cfg is None or M < BLOCK_M:
        return False
    row_tiles = -(-M // BLOCK_M)
    return row_tiles >= cfg.min_row_tiles or row_tiles in cfg.row_tiles


def gluon_gemm_aligned(*tensors: torch.Tensor) -> bool:
    """Whether the kernel can read these tensors: contiguous, and at an
    address Triton specializes as aligned."""
    return all(t.is_contiguous() and t.data_ptr() % _ALIGN == 0 for t in tensors)


def weight_scale_cols(weight_scale: torch.Tensor, N: int) -> torch.Tensor:
    """`[N / g, K / 32]` E8M0 weight scales (g = 1 or 32) as the
    `[K / 32, N]` uint8 tensor `rocm_mxfp8_gluon_gemm` reads."""
    scale = weight_scale.view(torch.uint8)
    return scale.repeat_interleave(N // scale.shape[0], dim=0).T.contiguous()


def rocm_mxfp8_gluon_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    weight_scale_cols: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """`out[...] = x @ weight.T` for a tuned `(N, K)`.

    Args:
        x: [M, K] e4m3 activation, M >= 256; with `x_scale` and the weight
            tensors, `gluon_gemm_aligned`.
        x_scale: [M, K / 32] E8M0 (uint8) activation scales.
        weight_preshuffled: [N, K] e4m3 weight in AITER's (16, 16) layout.
        weight_scale_cols: [K / 32, N] uint8, see `weight_scale_cols`.
        out: [M, N] bf16 output, contiguous.

    """
    M, K = x.shape
    N = weight_preshuffled.shape[0]
    cfg = _TUNED[(N, K)]
    assert M >= BLOCK_M and N % BLOCK_N == 0
    assert K % (2 * BLOCK_K) == 0 and K >= 4 * BLOCK_K and N * K < 1 << 31
    assert x_scale.shape == (M, K // 32) and weight_scale_cols.shape == (K // 32, N)
    assert gluon_gemm_aligned(x, x_scale, weight_preshuffled, weight_scale_cols)
    assert out.is_contiguous()
    nm, nn = -(-M // BLOCK_M), N // BLOCK_N
    nn_full = _full_column_tiles(nm, nn)
    nn_half = 2 * (nn - nn_full)
    _mxfp8_gluon_gemm_kernel[(nm * (nn_full + nn_half),)](
        x.view(torch.float8_e4m3fn),
        weight_preshuffled.view(torch.float8_e4m3fn),
        out,
        x_scale.view(torch.uint8),
        weight_scale_cols,
        M,
        K,
        N,
        K // 32 // 4,
        N // 4,
        nm,
        nn_full,
        nn_half,
        MFMA32=cfg.mfma32,
        GROUP_M=cfg.group_m,
        GROUP_N=cfg.group_n,
        num_warps=_NUM_WARPS,
        # Accumulators stay in VGPRs: two waves fit a SIMD without AGPR copies.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
