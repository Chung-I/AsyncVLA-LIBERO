r"""EdgeArch: single source of truth for the AsyncVLA-LIBERO edge adapter's capacity.

The edge adapter (`Edge_adapter_manip`) and its companion action-token projector
(`Proj_Actiontokens`) are currently trained at 512 width / 2 heads / 2 layers (the class
defaults). The ORIGINAL AsyncVLA runs the edge at 1024 width / 4 heads / 4 layers
(`config_nav/dataset_config.yaml`). Heads/layers are NOT recoverable from a `state_dict`'s
tensor shapes alone (attention block shapes depend on `embed_dim`, but the *number* of
layers/heads is architecture, not a tensor shape you can infer post hoc) -- so an eval that
rebuilds the edge with the wrong `mha_num_attention_heads`/`mha_num_attention_layers` will
either fail to load or silently load garbage into mismatched positions.

This module is the ONE place both `vla-scripts/train_asyncvla_libero.py` (which trains and
saves `edge_arch.json` next to each checkpoint) and `experiments/robot/libero/edge_policy.py`
(which loads `edge_arch.json` to rebuild the exact same architecture before `load_state_dict`)
import from, so the two code paths cannot drift.

Backward compatibility (MANDATORY): checkpoints saved before this module existed (all
current Phase-1 checkpoints and any run already in flight) have NO `edge_arch.json` next to
them. `load_edge_arch` treats that as "trained at the historical defaults" and returns
`EdgeArch()` (512/2/2/4) -- so old checkpoints keep loading exactly as they did before this
change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Union

import torch

from prismatic.models.small_head import Edge_adapter_manip, Proj_Actiontokens

EDGE_ARCH_FILENAME = "edge_arch.json"


@dataclass
class EdgeArch:
    """Edge adapter capacity. Defaults (512/2/2/4) match the historical hardcoded values
    used by every Phase-1 checkpoint trained before this module existed."""

    obs_encoding_size: int = 512
    mha_num_attention_heads: int = 2
    mha_num_attention_layers: int = 2
    mha_ff_dim_factor: int = 4


def build_edge_and_proj(
    llm_dim: int, device: torch.device, arch: EdgeArch = EdgeArch()
) -> Tuple[Edge_adapter_manip, Proj_Actiontokens]:
    """Builds the trainable edge adapter + action-token projector (fp32), on `device`, at
    the given `arch`.

    INVARIANT: `Proj_Actiontokens(action_dim=...)` MUST equal
    `Edge_adapter_manip(obs_encoding_size=...)` -- the projector's output tokens are
    concatenated with the edge's own image-encoding tokens (see `Edge_adapter_manip.forward`),
    so their per-token feature width must match. Both are derived from `arch.obs_encoding_size`
    here to keep that invariant true by construction.
    """
    proj = Proj_Actiontokens(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=arch.obs_encoding_size)
    edge = Edge_adapter_manip(
        obs_encoding_size=arch.obs_encoding_size,
        mha_num_attention_heads=arch.mha_num_attention_heads,
        mha_num_attention_layers=arch.mha_num_attention_layers,
        mha_ff_dim_factor=arch.mha_ff_dim_factor,
    )
    proj = proj.to(device=device, dtype=torch.float32)
    edge = edge.to(device=device, dtype=torch.float32)
    return edge, proj


def save_edge_arch(run_dir: Union[str, Path], arch: EdgeArch) -> None:
    """Writes `edge_arch.json` into `run_dir` (created if missing), recording the edge
    architecture used for every checkpoint saved in that directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / EDGE_ARCH_FILENAME, "w") as f:
        json.dump(asdict(arch), f, indent=2)


def load_edge_arch(ckpt_path: Union[str, Path]) -> EdgeArch:
    """Reads `edge_arch.json` from `ckpt_path`'s PARENT directory (i.e. the run directory a
    checkpoint file like `shead--50000_checkpoint.pt` lives in). Falls back to `EdgeArch()`
    (512/2/2/4) if the file is absent -- BACKWARD COMPAT for checkpoints saved before
    `edge_arch.json` existed.
    """
    arch_path = Path(ckpt_path).parent / EDGE_ARCH_FILENAME
    if not arch_path.exists():
        return EdgeArch()
    with open(arch_path, "r") as f:
        data = json.load(f)
    return EdgeArch(**data)
