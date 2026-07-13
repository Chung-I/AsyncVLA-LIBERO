r"""Latency microbenchmark for the async-VLA base/edge split.

Measures, on the real frozen OpenVLA-OFT LIBERO base + the small edge adapter:

  - ``t_base_ms``: mean wall-clock of ONE base forward pass
    (`extract_actions_hidden_states`) + hidden-state projection
    (`Proj_Actiontokens.predict_action`) -- the expensive "server-side" step that
    only needs to run every N control steps in the async scheme.
  - ``t_edge_ms``: mean wall-clock of ONE `Edge_adapter` forward -- the cheap
    "client-side" step that runs every control step.

Both timings synchronize the GPU immediately before and after each timed region
(`torch.cuda.synchronize()`), since CUDA kernel launches are asynchronous and
`time.perf_counter()` alone would only measure host-side dispatch time, not actual
GPU compute.

Given these two numbers, the analytic per-step compute cost of the async scheme
running the (expensive) base every N steps and the (cheap) edge every step is
``t_edge + t_base / N``, and the speedup vs. running the base every step
(the synchronous/vanilla baseline) is ``t_base / (t_edge + t_base / N)``.

Usage:
    .venv/bin/python -m experiments.robot.libero.latency_bench
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Optional

import torch

from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
from experiments.robot.libero.base_features import extract_actions_hidden_states
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj
from prismatic.vla.datasets.libero_dataset import make_dummy_base_batch


def benchmark(
    n_warmup: int = 5,
    n_iter: int = 50,
    device: Optional[torch.device] = None,
    edge_arch: EdgeArch = EdgeArch(),
) -> Dict[str, float]:
    """Measure mean per-call latency of the base (forward + projection) and the edge.

    Args:
        n_warmup: number of iterations to run and discard before timing starts (lets
            CUDA kernels JIT-compile/cache and lets clocks/allocators settle).
        n_iter: number of timed iterations to average over.
        device: device to run on; defaults to the frozen base's device (i.e. wherever
            `build_frozen_base` placed it -- typically `cuda:0`).
        edge_arch: edge adapter capacity to bench (see `experiments.robot.libero.edge_arch.
            EdgeArch`); defaults to the historical 512/2/2/4 Phase-1 capacity.

    Returns:
        `{"t_base_ms": float, "t_edge_ms": float, "n_iter": int}`.
    """
    vla, processor, proprio_projector = build_frozen_base(LiberoBaseConfig())
    if device is None:
        device = next(vla.parameters()).device

    batch, num_patches = make_dummy_base_batch(processor)

    edge, proj = build_edge_and_proj(vla.llm_dim, device, edge_arch)
    proj.eval()
    edge.eval()

    batch_size = batch["input_ids"].shape[0]
    taskid = torch.zeros(batch_size, device=device)
    obs96 = torch.randn(batch_size, 3, 96, 96, device=device)
    past96 = torch.randn(batch_size, 3, 96, 96, device=device)

    total_iters = n_warmup + n_iter
    base_times = []
    edge_times = []

    with torch.no_grad():
        for i in range(total_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            hs = extract_actions_hidden_states(vla, batch, proprio_projector, num_patches, device=device)
            vla_feature = proj.predict_action(hs.to(torch.float32), taskid)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            edge(obs96, past96, vla_feature)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            if i >= n_warmup:
                base_times.append(t1 - t0)
                edge_times.append(t3 - t2)

    t_base_ms = 1000.0 * sum(base_times) / len(base_times)
    t_edge_ms = 1000.0 * sum(edge_times) / len(edge_times)

    return {"t_base_ms": t_base_ms, "t_edge_ms": t_edge_ms, "n_iter": n_iter}


def _print_speedup_table(t_base_ms: float, t_edge_ms: float) -> None:
    print(f"\n{'N':>4}  {'per-step compute (ms)':>22}  {'speedup vs base-every-step':>26}")
    for n in (1, 2, 4, 8, 16):
        per_step = t_edge_ms + t_base_ms / n
        speedup = t_base_ms / per_step
        print(f"{n:>4}  {per_step:>22.3f}  {speedup:>26.2f}x")


def _parse_args() -> argparse.Namespace:
    # Flag defaults are the FAITHFUL edge capacity (1024/4/4/4, `config_nav/dataset_config.yaml`),
    # not `EdgeArch()`'s 512/2/2/4 historical/back-compat class defaults -- running this bench
    # with no flags must report t_edge for the faithful curve, not a silently smaller edge.
    _faithful = EdgeArch.from_config_nav()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs_encoding_size", type=int, default=_faithful.obs_encoding_size,
                         help="Edge adapter token width (Edge_adapter.obs_encoding_size).")
    parser.add_argument("--heads", type=int, default=_faithful.mha_num_attention_heads,
                         help="Edge adapter MHA attention heads (Edge_adapter.mha_num_attention_heads).")
    parser.add_argument("--layers", type=int, default=_faithful.mha_num_attention_layers,
                         help="Edge adapter MHA attention layers (Edge_adapter.mha_num_attention_layers).")
    parser.add_argument("--ff_dim_factor", type=int, default=_faithful.mha_ff_dim_factor,
                         help="Edge adapter MHA feed-forward dim factor (Edge_adapter.mha_ff_dim_factor).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    edge_arch = EdgeArch(
        obs_encoding_size=args.obs_encoding_size,
        mha_num_attention_heads=args.heads,
        mha_num_attention_layers=args.layers,
        mha_ff_dim_factor=args.ff_dim_factor,
    )
    result = benchmark(edge_arch=edge_arch)
    print(result)
    _print_speedup_table(result["t_base_ms"], result["t_edge_ms"])
