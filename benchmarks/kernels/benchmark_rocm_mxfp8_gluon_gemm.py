# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the Gluon MXFP8 GEMM against AITER's FlyDSL GEMM on gfx950.

Both read the same (16, 16)-preshuffled weights. Each timing is a HIP graph
replay over rotating weights (> 400 MiB, so they are read cold) timed with
device events; the two GEMMs alternate order over `--repeats`, and a row
reports the median and the spread of the repeats. Outputs are compared
before timing.

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
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import _aiter_gemm

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
    fp8 = torch.float8_e4m3fn
    print(torch.cuda.get_device_name(0))
    print("N K M flydsl_us gluon_us gluon/flydsl spread tflops routed max_diff")
    for n, k in rocm_gluon_gemm._TUNED:
        weights = []
        for _ in range(math.ceil(COLD_BYTES / (n * k))):
            weight = (torch.randn(n, k, device="cuda") * 2).to(fp8)
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
            x = (torch.randn(m, k, device="cuda") * 2).to(fp8)
            x_scale = torch.randint(
                120, 130, (m, k // 32), device="cuda", dtype=torch.uint8
            )
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            gluon = [
                partial(rocm_gluon_gemm.rocm_mxfp8_gluon_gemm, x, x_scale, w, cols, out)
                for w, _, cols in weights
            ]
            flydsl = [
                partial(_aiter_gemm, x, x_scale, w, scale, out)
                for w, scale, _ in weights
            ]
            max_diff = 0.0
            for gluon_call, flydsl_call in zip(gluon, flydsl):
                flydsl_call()
                expected = out.float()
                gluon_call()
                max_diff = max(max_diff, (out.float() - expected).abs().max().item())
            times: dict[str, list[float]] = {"flydsl": [], "gluon": []}
            for repeat in range(args.repeats):
                order = [("flydsl", flydsl), ("gluon", gluon)]
                for name, calls in order if repeat % 2 == 0 else order[::-1]:
                    times[name].append(time_graph_us(calls))
            t_fly = statistics.median(times["flydsl"])
            t_gluon = statistics.median(times["gluon"])
            spread = max(
                (max(t) - min(t)) / statistics.median(t) for t in times.values()
            )
            print(
                f"{n} {k} {m} {t_fly:.1f} {t_gluon:.1f} {t_gluon / t_fly:.3f} "
                f"{spread:.1%} {2 * m * n * k / t_gluon / 1e6:.0f} "
                f"{rocm_gluon_gemm.gluon_gemm_enabled(m, n, k)} {max_diff:.3g}",
                flush=True,
            )
        del weights


if __name__ == "__main__":
    main()
