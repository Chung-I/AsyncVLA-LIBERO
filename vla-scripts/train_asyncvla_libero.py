"""
train_asyncvla_libero.py

Train the AsyncVLA edge adapter + action-token projector on LIBERO-Spatial, on top of a
FROZEN OpenVLA-OFT LIBERO base checkpoint.

Structure mirrors `vla-scripts/train_asyncvla.py` (config dataclass, DDP init via
`accelerate.PartialState`, wandb init, AdamW optimizer, periodic checkpoint save,
`train/l1_loss` logging), but with LoRA and all navigation datasets/loss terms removed:

  - Base: `experiments.robot.libero.base_config.build_frozen_base` -> frozen
    `(vla, processor, proprio_projector)`. Never trained; no LoRA.
  - Per-step forward: `extract_actions_hidden_states` (base, no_grad) ->
    `proj.predict_action` -> `edge(obs_img_96, past_img_96, vla_feature)`.
  - Loss: `F.l1_loss(pred_chunk, gt_action_chunk)` on the [B, 8, 7] action chunk.
  - Trainable params: `Edge_adapter_manip` ("shead") + `Proj_Actiontokens` ("proj") only.

CLI entrypoint (torchrun-compatible):

    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_asyncvla_libero.py \
        --wandb_entity <entity> --wandb_project asyncvla-libero

`train_one_batch_smoke(num_iters)` is a separate, single-process/single-GPU entrypoint (no
torchrun/DDP required) that overfits ONE fixed real batch from `LiberoSpatialDataset` --
used by the `tests/test_train_overfit.py` slow integration test.
"""

# NOTE: do NOT add `from __future__ import annotations` here. It turns the
# `cfg: AsyncVLALiberoConfig` type hint into a string that draccus 0.8.0's
# `@draccus.wrap()` cannot resolve, so it calls `dataclasses.fields()` on the
# string and raises "must be called with a dataclass type or instance".

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
from experiments.robot.libero.base_features import extract_actions_hidden_states
# `build_edge_and_proj` is re-exported here (not just imported) so existing callers that do
# `from vla_scripts.train_asyncvla_libero import build_edge_and_proj`
# (`train_one_batch_smoke`, `tests/test_train_overfit*.py`, `latency_bench.py`-style usage)
# keep working unchanged; `edge_arch.py` is the single source of truth for its body.
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj, save_edge_arch
from prismatic.vla.datasets.libero_dataset import (
    NUM_BASE_PATCHES,
    LiberoSpatialDataset,
    collate_libero_batch,
)

# ==============================
# Config
# ==============================


@dataclass
class AsyncVLALiberoConfig:
    # fmt: off
    # Dataset
    data_root_dir: Optional[str] = None     # Local `modified_libero_rlds` dir; None -> dataset default
    split: str = "train"
    episode_limit: int = 10_000             # Max RLDS episodes to load (LIBERO-Spatial has ~432; 10k = all).
                                            #   NOTE: the dataset default is 2 (smoke-only) -- training MUST
                                            #   pass a large limit or it trains on 2 episodes.
    delay_aware: bool = False               # Phase-2 delay-aware training (base sees I_{t-k}).
    k_max: int = 15                         # Max delay k ~ Uniform{0..k_max} when delay_aware.

    # Training configuration
    batch_size: int = 8                     # Batch size per device
    learning_rate: float = 1e-4
    max_steps: int = 50_000
    save_freq: int = 5_000                  # Checkpoint saving frequency in steps
    num_workers: int = 0                    # 0 = load in the main process. num_workers>0 forks AFTER the
                                            #   frozen base + TensorFlow are initialized, which deadlocks
                                            #   the DataLoader workers (classic fork-after-CUDA/TF hang).

    run_root_dir: Path = Path("runs_libero")  # Path to directory to store checkpoints
    run_id_note: Optional[str] = None

    # Edge adapter architecture (see `experiments.robot.libero.edge_arch.EdgeArch`). Defaults
    # (512/2/2/4) match the historical hardcoded Phase-1 capacity; the ORIGINAL AsyncVLA runs
    # the edge at 1024/4/4/4. `edge_arch.json` is saved alongside every checkpoint so eval can
    # auto-detect which capacity a given checkpoint was trained at.
    edge_obs_encoding_size: int = 512
    edge_mha_heads: int = 2
    edge_mha_layers: int = 2
    edge_mha_ff_dim_factor: int = 4

    # Logging
    wandb_entity: Optional[str] = None
    wandb_project: str = "asyncvla-libero"
    wandb_log_freq: int = 10
    # fmt: on


# ==============================
# Helpers
# ==============================


def get_run_id(cfg: AsyncVLALiberoConfig) -> str:
    run_id = f"asyncvla_libero+b{cfg.batch_size}+lr-{cfg.learning_rate}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def save_training_checkpoint(run_dir: Path, step: int, edge: nn.Module, proj: nn.Module, arch: EdgeArch) -> None:
    """Saves `shead--<step>_checkpoint.pt` (edge adapter) and `proj--<step>_checkpoint.pt`
    (action-token projector). Base model / proprio projector are frozen and never saved.
    Also (re)writes `edge_arch.json` into `run_dir` so eval can auto-detect the architecture
    these checkpoints were trained with (see `experiments.robot.libero.edge_arch`)."""
    os.makedirs(run_dir, exist_ok=True)
    torch.save(unwrap(edge).state_dict(), run_dir / f"shead--{step}_checkpoint.pt")
    torch.save(unwrap(proj).state_dict(), run_dir / f"proj--{step}_checkpoint.pt")
    save_edge_arch(run_dir, arch)
    print(f"Saved checkpoint at step {step} to {run_dir}")


def run_forward_pass(
    vla,
    proprio_projector,
    edge: nn.Module,
    proj: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-step forward: frozen base (no_grad) -> proj -> edge -> L1 chunk loss.

    Returns (loss, metrics) where `loss` carries gradients (for edge + proj only) and
    `metrics` holds detached values for logging.
    """
    batch_size = batch["input_ids"].shape[0]
    taskid = torch.zeros(batch_size, device=device)  # constant modality id for LIBERO (single-modality)

    # Frozen base forward (no_grad internally) -> vla_feature source.
    hidden = extract_actions_hidden_states(
        vla, batch, proprio_projector, num_patches=NUM_BASE_PATCHES, device=device
    )  # [B, 56, 4096] bfloat16, no grad

    vla_feature = proj.predict_action(hidden.to(torch.float32), taskid)  # [B, 8, 512] fp32

    obs_img_96 = batch["obs_img_96"].to(device=device, dtype=torch.float32)
    past_img_96 = batch["past_img_96"].to(device=device, dtype=torch.float32)
    pred_chunk = edge(obs_img_96, past_img_96, vla_feature)  # [B, 8, 7] fp32

    gt_action_chunk = batch["gt_action_chunk"].to(device=device, dtype=torch.float32)  # [B, 8, 7]
    loss = F.l1_loss(pred_chunk, gt_action_chunk)

    metrics = {"l1_loss": loss.item()}
    return loss, metrics


# ==============================
# Main torchrun entrypoint
# ==============================


@draccus.wrap()
def train_asyncvla_libero(cfg: AsyncVLALiberoConfig) -> None:
    """Trains the edge adapter + projector on LIBERO-Spatial. Base + proprio projector are
    frozen throughout. `torchrun`-compatible (uses `accelerate.PartialState` for DDP setup,
    same pattern as `vla-scripts/train_asyncvla.py`)."""

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id

    # GPU / DDP setup.
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_id}")
    print("World size", world_size, "rank", device_id)

    if distributed_state.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Load frozen base (no LoRA, no training of the base).
    vla, processor, proprio_projector = build_frozen_base(LiberoBaseConfig())
    print("vla class", type(vla), "llm_dim", vla.llm_dim)

    # Build trainable edge adapter + projector.
    edge_arch = EdgeArch(
        obs_encoding_size=cfg.edge_obs_encoding_size,
        mha_num_attention_heads=cfg.edge_mha_heads,
        mha_num_attention_layers=cfg.edge_mha_layers,
        mha_ff_dim_factor=cfg.edge_mha_ff_dim_factor,
    )
    edge, proj = build_edge_and_proj(vla.llm_dim, device, edge_arch)
    count_parameters(edge, "edge (shead)")
    count_parameters(proj, "proj")

    if world_size > 1:
        edge = wrap_ddp(edge, device_id, find_unused=True)
        proj = wrap_ddp(proj, device_id, find_unused=True)

    trainable_params = list(edge.parameters()) + list(proj.parameters())
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Dataset / dataloader.
    dataset = LiberoSpatialDataset(
        split=cfg.split, data_dir=cfg.data_root_dir, processor=processor, episode_limit=cfg.episode_limit,
        delay_aware=cfg.delay_aware, k_max=cfg.k_max
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=device_id, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    def collate_fn(instances):
        return collate_libero_batch(instances, pad_token_id=processor.tokenizer.pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    recent_l1 = deque(maxlen=cfg.wandb_log_freq)
    edge.train()
    proj.train()

    step = 0
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        for epoch in range(10_000):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if step >= cfg.max_steps:
                    break

                loss, metrics = run_forward_pass(vla, proprio_projector, edge, proj, batch, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                recent_l1.append(metrics["l1_loss"])
                progress.update()

                if distributed_state.is_main_process and step % cfg.wandb_log_freq == 0:
                    wandb.log({"train/l1_loss": sum(recent_l1) / len(recent_l1)}, step=step)

                if step > 0 and step % cfg.save_freq == 0:
                    if world_size > 1:
                        dist.barrier()
                    if distributed_state.is_main_process:
                        save_training_checkpoint(run_dir, step, edge, proj, edge_arch)
                    if world_size > 1:
                        dist.barrier()

                step += 1
            if step >= cfg.max_steps:
                break

    if distributed_state.is_main_process:
        save_training_checkpoint(run_dir, step, edge, proj, edge_arch)


# ==============================
# Single-process overfit smoke test
# ==============================


def train_one_batch_smoke(num_iters: int = 200, delay_aware: bool = False, k_max: int = 15) -> List[float]:
    """Loads the frozen base ONCE, builds ONE fixed real batch from `LiberoSpatialDataset`
    (local shard), and runs `num_iters` forward/backward steps of edge+proj on that single
    batch. Returns the per-step L1 loss list.

    Single-process, single-GPU: does NOT use `torchrun`/DDP/`accelerate.PartialState`, and
    does NOT touch wandb. Used by `tests/test_train_overfit.py`.

    Robustness comes from the iteration count, not a cherry-picked seed: a 3-seed sweep
    under deterministic cuDNN gives final/initial L1 ratios of ~0.51-0.55 at 50 iters
    (flaky vs the 0.5 bar), ~0.10-0.13 at 150 iters, and ~0.03-0.05 at 300 iters. The
    200-iter default clears the "< 0.5x initial loss" bar with a wide margin for any seed;
    the fixed `torch.manual_seed` below only makes the trajectory reproducible run-to-run.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Reproducible overfit trajectory (see docstring).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Frozen base — loaded once.
    vla, processor, proprio_projector = build_frozen_base(LiberoBaseConfig())

    # ONE fixed real batch.
    dataset = LiberoSpatialDataset(split="train", max_samples=1, processor=processor,
                                   delay_aware=delay_aware, k_max=k_max)
    item = dataset[0]
    batch = collate_libero_batch([item], pad_token_id=processor.tokenizer.pad_token_id)

    # Trainable edge + projector. Seed here (last RNG-touching call before init) for a
    # reproducible init.
    torch.manual_seed(21)
    edge, proj = build_edge_and_proj(vla.llm_dim, device)
    optimizer = AdamW(list(edge.parameters()) + list(proj.parameters()), lr=1e-4)

    edge.train()
    proj.train()

    losses: List[float] = []
    for _ in range(num_iters):
        loss, metrics = run_forward_pass(vla, proprio_projector, edge, proj, batch, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(metrics["l1_loss"])

    return losses


if __name__ == "__main__":
    train_asyncvla_libero()
