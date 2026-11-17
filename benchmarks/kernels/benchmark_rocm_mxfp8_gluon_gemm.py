# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the Gluon MXFP8 GEMM against AITER's FlyDSL GEMM on gfx950.

Both read the same (16, 16)-preshuffled weight. A row is one M of a tuned
`(N, K)`: the best per-call microseconds of each GEMM, their ratio, whether
`gluon_gemm_enabled` picks the Gluon GEMM there, and the largest difference
between the two outputs.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_rocm_mxfp8_gluon_gemm.py \
        --m 256 1024 4096 8192 8300

"""

import argparse
import time
from functools import partial

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.mxfp8 import rocm_gluon_gemm
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import _aiter_gemm

DEFAULT_MS = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6000, 8192, 8300, 16384]


def best_us(fn, calls: int, rounds: int) -> float:
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        torch.accelerator.synchronize()
        best = min(best, (time.perf_counter() - start) / calls)
    return best * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_MS)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    fp8 = torch.float8_e4m3fn
    for n, k in rocm_gluon_gemm._TUNED:
        weight = (torch.randn(n, k, device="cuda") * 2).to(fp8)
        weight_scale = torch.randint(
            120, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8
        )
        shuffled = rocm_aiter_ops.shuffle_weight(weight, (16, 16))
        scale_cols = rocm_gluon_gemm.weight_scale_cols(weight_scale, n)
        for m in args.m:
            x = (torch.randn(m, k, device="cuda") * 2).to(fp8)
            x_scale = torch.randint(
                120, 130, (m, k // 32), device="cuda", dtype=torch.uint8
            )
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ref = torch.empty_like(out)

            gluon = partial(
                rocm_gluon_gemm.rocm_mxfp8_gluon_gemm,
                x,
                x_scale,
                shuffled,
                scale_cols,
                out,
            )
            flydsl = partial(_aiter_gemm, x, x_scale, shuffled, weight_scale, ref)
            t_fly = best_us(flydsl, args.calls, args.rounds)
            t_gluon = best_us(gluon, args.calls, args.rounds)
            diff = (out.float() - ref.float()).abs().max().item()
            print(
                f"N={n} K={k} M={m}: flydsl {t_fly:.1f} us, gluon {t_gluon:.1f} us "
                f"({t_gluon / t_fly:.3f}x), "
                f"enabled={rocm_gluon_gemm.gluon_gemm_enabled(m, n, k)}, "
                f"max diff {diff:.3g}"
            )


if __name__ == "__main__":
    main()
