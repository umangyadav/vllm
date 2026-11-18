# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the Gluon MXFP8 GEMM against AITER's FlyDSL GEMM on gfx950.

Both read the same (16, 16)-preshuffled weights. Each timing is a HIP graph
replay over rotating weights (> 400 MiB, so they are read cold) timed with
device events; the variants alternate order over `--repeats`, and a row
reports their medians and the largest spread. Before timing, each Gluon
variant must reproduce FlyDSL's output in a buffer filled with NaN.

Kernel columns take a quantized activation: `flydsl`, `rows` (Gluon,
row-major activation scales; whole tiles of 256 rows only, else nan) and
`cols` (Gluon, transposed scales, rows padded to whole tiles). The `op`
columns time `rocm_mxfp8_block32_gemm_quant` on a bf16 activation, quantizer
included, without and with the Gluon route.

Example (record the commit):
    git rev-parse HEAD
    HIP_FORCE_DEV_KERNARG=1 .venv/bin/python \
        benchmarks/kernels/benchmark_rocm_mxfp8_gluon_gemm.py --m 2048 8192 8300

"""

import argparse
import math
import statistics
from functools import partial

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.mxfp8 import rocm_gluon_gemm
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (
    _aiter_gemm,
    rocm_mxfp8_block32_gemm_quant,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_triton,
)

COLD_BYTES = 400 * 1024**2
INNER = 2
REPLAYS = 10
ROUNDS = 5
WARMUP = 5
DEFAULT_MS = [256, 300, 512, 768, 1024, 1536, 2048, 3072, 4096, 4103, 6144, 8192]


def time_graph_us(calls) -> float:
    """Median microseconds per call of one graph cycling over `calls`."""
    for call in calls:
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(len(calls) * INNER):
            calls[i % len(calls)]()
    torch.accelerator.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(ROUNDS)]
    for _ in range(WARMUP):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        end.record()
    torch.accelerator.synchronize()
    per_call = REPLAYS * len(calls) * INNER
    return statistics.median(s.elapsed_time(e) / per_call for s, e in events) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_MS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0)
    bf16 = torch.bfloat16
    print(torch.cuda.get_device_name(0))
    print(
        "N K M flydsl rows cols rows/flydsl cols/flydsl op_flydsl op_gluon "
        "op_ratio spread routed_rows routed_cols max_diff"
    )
    for n, k in rocm_gluon_gemm._TUNED:
        weights = []
        for _ in range(math.ceil(COLD_BYTES / (n * k))):
            weight = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
            scale = torch.randint(
                120, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8
            )
            weights.append(
                (
                    rocm_aiter_ops.shuffle_weight(weight, (16, 16)),
                    scale,
                    rocm_gluon_gemm.weight_scale_cols(scale, n),
                )
            )
        for m in args.m:
            x = torch.randn(m, k, device="cuda", dtype=bf16)
            x_q, x_scale = _mxfp8_e4m3_quantize_triton(x)
            rows = -(-m // 256) * 256
            x_qp, x_scale_cols = _mxfp8_e4m3_quantize_triton(x, rows)
            out = torch.empty(rows, n, device="cuda", dtype=bf16)
            gluon = rocm_gluon_gemm.rocm_mxfp8_gluon_gemm
            variants = {
                "flydsl": [
                    partial(_aiter_gemm, x_q, x_scale, w, scale, out[:m])
                    for w, scale, _ in weights
                ],
                "rows": [
                    partial(gluon, x_q, x_scale, w, cols, out[:m])
                    for w, _, cols in weights
                ],
                "cols": [
                    partial(gluon, x_qp, x_scale_cols, w, cols, out, True)
                    for w, _, cols in weights
                ],
                "op_flydsl": [
                    partial(rocm_mxfp8_block32_gemm_quant, x, w, scale, bf16, w)
                    for w, scale, _ in weights
                ],
                "op_gluon": [
                    partial(rocm_mxfp8_block32_gemm_quant, x, w, scale, bf16, w, cols)
                    for w, scale, cols in weights
                ],
            }
            if m % 256:
                del variants["rows"]
            variants["flydsl"][0]()
            expected = out[:m].float()
            max_diff = 0.0
            for name in set(variants) & {"rows", "cols", "op_gluon"}:
                out.fill_(float("nan"))
                result = variants[name][0]()
                diff = (out[:m] if result is None else result).float() - expected
                if not diff.norm() < 1e-3 * expected.norm():
                    raise RuntimeError(f"{name} disagrees with flydsl at M={m}")
                max_diff = max(max_diff, diff.abs().max().item())
            times: dict[str, list[float]] = {name: [] for name in variants}
            for repeat in range(args.repeats):
                order = list(variants.items())
                for name, calls in order if repeat % 2 == 0 else order[::-1]:
                    times[name].append(time_graph_us(calls))
            t = {name: statistics.median(v) for name, v in times.items()}
            t.setdefault("rows", math.nan)
            spread = max((max(v) - min(v)) / t[name] for name, v in times.items())
            enabled = rocm_gluon_gemm.gluon_gemm_enabled
            print(
                f"{n} {k} {m} {t['flydsl']:.1f} {t['rows']:.1f} {t['cols']:.1f} "
                f"{t['rows'] / t['flydsl']:.3f} {t['cols'] / t['flydsl']:.3f} "
                f"{t['op_flydsl']:.1f} {t['op_gluon']:.1f} "
                f"{t['op_gluon'] / t['op_flydsl']:.3f} {spread:.1%} "
                f"{enabled(m, n, k)} {enabled(m, n, k, True)} {max_diff:.3g}",
                flush=True,
            )
        del weights


if __name__ == "__main__":
    main()
