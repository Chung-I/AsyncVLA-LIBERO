r"""EdgeArch: single source of truth for the AsyncVLA-LIBERO edge adapter's capacity.

The edge adapter is the ORIGINAL `Edge_adapter` class (`prismatic.models.small_head`), with one
added constructor kwarg -- `action_dim: int = 4` -- so its output head can be built at either the
original nav width (the default, 4, used by every original caller) or the LIBERO width (7, passed
explicitly here via `arch`/`ACTION_DIM`). This is NOT a no-op under `ACTION_DIM`: this branch
retargeted that constant from 4 (nav) to 7 (LIBERO) in `prismatic/vla/constants.py` (commit
`7954582`), so the head width must be parameterized rather than derived from `ACTION_DIM` directly
-- see `tests/test_faithful_edge.py` for the full rationale. Plus its companion
action-token projector (`Proj_Actiontokens`). Historical AsyncVLA-LIBERO runs trained it at
512 width / 2 heads / 2 layers (the class defaults). The ORIGINAL AsyncVLA runs the edge at
1024 width / 4 heads / 4 layers (`config_nav/dataset_config.yaml`, readable via
`EdgeArch.from_config_nav()`). Heads/layers are NOT recoverable from a `state_dict`'s tensor
shapes alone (attention block shapes depend on `embed_dim`, but the *number* of layers/heads
is architecture, not a tensor shape you can infer post hoc) -- so an eval that rebuilds the
edge with the wrong `mha_num_attention_heads`/`mha_num_attention_layers` will either fail to
load or silently load garbage into mismatched positions.

This module is the ONE place both `vla-scripts/train_asyncvla_libero.py` (which trains and
saves `edge_arch.json` next to each checkpoint) and `experiments/robot/libero/edge_policy.py`
(which loads `edge_arch.json` to rebuild the exact same architecture before `load_state_dict`)
import from, so the two code paths cannot drift.

No silent fallback (MANDATORY): `mha_num_attention_heads` is the ONE arch field that leaves
NO fingerprint in the weights -- `nn.MultiheadAttention` stores `in_proj_weight` as
`[3*embed_dim, embed_dim]`, a shape independent of `num_heads`. A checkpoint trained at 4
heads loads CLEANLY (no shape error) into a module built with 2 heads, and silently computes
attention with the wrong head split -- no exception, no NaN, just quietly wrong outputs. So
`load_edge_arch` does NOT guess a default when `edge_arch.json` is missing: it raises,
telling the caller to pass the arch explicitly (`load_edge_arch(ckpt_path, arch=...)`).
`validate_edge_arch` closes the other half of the gap: it cross-checks an `EdgeArch` against
what the checkpoint's weights DO reveal (`obs_encoding_size`, transformer layer count) and
raises loudly on a mismatch, so `EdgePolicy` fails at load time rather than serving garbage.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
from prismatic.vla.constants import ACTION_DIM

EDGE_ARCH_FILENAME = "edge_arch.json"

_SA_LAYER_INDEX_RE = re.compile(r"^decoder\.sa_decoder\.layers\.(\d+)\.")


@dataclass
class EdgeArch:
    """Edge adapter capacity. Defaults (512/2/2/4) match the historical hardcoded values
    used by every Phase-1 checkpoint trained before this module existed.

    DELIBERATELY LEFT AS-IS: kept as the documented historical Phase-1 capacity, and as a
    convenient literal for tests -- do NOT change them to the faithful 1024/4/4/4 capacity
    just because `build_edge_and_proj`'s `arch` parameter below became required; that
    argument is unrelated to this class's own field defaults. NOTE: `load_edge_arch` no
    longer falls back to these defaults when `edge_arch.json` is missing (it raises
    instead, see that function's docstring) -- the head count cannot be safely guessed."""

    obs_encoding_size: int = 512
    mha_num_attention_heads: int = 2
    mha_num_attention_layers: int = 2
    mha_ff_dim_factor: int = 4

    @classmethod
    def from_config_nav(cls, path: str = "config_nav/dataset_config.yaml") -> "EdgeArch":
        """The ORIGINAL AsyncVLA edge capacity, read from the same file `train_asyncvla.py`
        reads (`config_nav/dataset_config.yaml`): 1024 / 4 heads / 4 layers / ff x4."""
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path(__file__).resolve().parents[3].joinpath(path).read_text())
        return cls(
            obs_encoding_size=int(cfg["obs_encoding_size"]),
            mha_num_attention_heads=int(cfg["mha_num_attention_heads"]),
            mha_num_attention_layers=int(cfg["mha_num_attention_layers"]),
            mha_ff_dim_factor=int(cfg["mha_ff_dim_factor"]),
        )


def build_edge_and_proj(
    llm_dim: int, device: torch.device, arch: EdgeArch
) -> Tuple[Edge_adapter, Proj_Actiontokens]:
    """Builds the trainable edge adapter + action-token projector (fp32), on `device`, at
    the given `arch`.

    `arch` is REQUIRED (no default): the class-field defaults on `EdgeArch` (512/2/2/4) are
    the historical Phase-1 *capacity*, kept only for `load_edge_arch`'s checkpoint back-compat
    (see that function's docstring) -- they must never be a silent default for a fresh build.
    Callers that want the faithful capacity must say so explicitly, e.g.
    `EdgeArch.from_config_nav()`.

    INVARIANT: `Proj_Actiontokens(action_dim=...)` MUST equal
    `Edge_adapter(obs_encoding_size=...)` -- the projector's output tokens are
    concatenated with the edge's own image-encoding tokens (see `Edge_adapter.forward`),
    so their per-token feature width must match. Both are derived from `arch.obs_encoding_size`
    here to keep that invariant true by construction.

    The edge's own output-head width (`NUM_ACTIONS_CHUNK * action_dim`) is set to the
    LIBERO `ACTION_DIM` (7) explicitly -- `Edge_adapter`'s own `action_dim` default (4) is the
    ORIGINAL navigation width and must stay untouched for every original caller.
    """
    proj = Proj_Actiontokens(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=arch.obs_encoding_size)
    edge = Edge_adapter(
        obs_encoding_size=arch.obs_encoding_size,
        mha_num_attention_heads=arch.mha_num_attention_heads,
        mha_num_attention_layers=arch.mha_num_attention_layers,
        mha_ff_dim_factor=arch.mha_ff_dim_factor,
        action_dim=ACTION_DIM,
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


def load_edge_arch(ckpt_path: Union[str, Path], arch: Optional[EdgeArch] = None) -> EdgeArch:
    """Reads `edge_arch.json` from `ckpt_path`'s PARENT directory (i.e. the run directory a
    checkpoint file like `shead--50000_checkpoint.pt` lives in).

    If `arch` is given explicitly, it is returned as-is and the json is never consulted --
    the escape hatch for callers who already know the correct architecture.

    Otherwise, RAISES `FileNotFoundError` if `edge_arch.json` is absent, rather than
    silently falling back to `EdgeArch()`'s 512/2/2/4 defaults. `mha_num_attention_heads`
    cannot be recovered from a checkpoint's tensor shapes (`nn.MultiheadAttention` stores
    `in_proj_weight` as `[3*embed_dim, embed_dim]`, independent of `num_heads`), so a wrong
    guess would load the checkpoint's weights CLEANLY -- no shape error -- while silently
    computing attention with the wrong head split. Guessing is therefore strictly worse than
    failing loudly here.
    """
    if arch is not None:
        return arch
    arch_path = Path(ckpt_path).parent / EDGE_ARCH_FILENAME
    if not arch_path.exists():
        raise FileNotFoundError(
            f"No {EDGE_ARCH_FILENAME} found next to checkpoint {str(ckpt_path)!r} (expected "
            f"at {str(arch_path)!r}). The edge adapter's head count "
            "(`mha_num_attention_heads`) cannot be recovered from the checkpoint's tensor "
            "shapes, so it cannot be safely guessed. Pass the correct architecture "
            "explicitly, e.g. `load_edge_arch(ckpt_path, arch=EdgeArch(...))`."
        )
    with open(arch_path, "r") as f:
        data = json.load(f)
    return EdgeArch(**data)


def validate_edge_arch(arch: EdgeArch, state_dict: Dict[str, torch.Tensor]) -> None:
    """Cross-checks `arch` against what `state_dict` (an `Edge_adapter.state_dict()`) DOES
    reveal, raising `ValueError` on a mismatch. Does NOT (cannot) check
    `mha_num_attention_heads` -- see the module docstring -- so a caller must obtain `arch`
    from a trustworthy source (`edge_arch.json`, written at save time by the same run that
    produced the checkpoint) rather than guessing.

    Checks:
      - `embed_dim`, inferred from a `decoder.sa_decoder.layers.*.self_attn.in_proj_weight`
        shape (`[3*embed_dim, embed_dim]` -> `embed_dim = shape[1]`), must equal
        `arch.obs_encoding_size`.
      - the number of transformer layers, inferred by counting distinct
        `decoder.sa_decoder.layers.<N>.` indices present in `state_dict`, must equal
        `arch.mha_num_attention_layers`.
    """
    layer_indices = {
        int(m.group(1)) for k in state_dict if (m := _SA_LAYER_INDEX_RE.match(k)) is not None
    }
    if not layer_indices:
        raise ValueError(
            "state_dict has no 'decoder.sa_decoder.layers.<N>.*' keys -- is this really an "
            "Edge_adapter state_dict?"
        )

    in_proj_key = f"decoder.sa_decoder.layers.{min(layer_indices)}.self_attn.in_proj_weight"
    embed_dim = state_dict[in_proj_key].shape[1]
    if embed_dim != arch.obs_encoding_size:
        raise ValueError(
            f"edge_arch mismatch: state_dict's {in_proj_key!r} shape "
            f"{tuple(state_dict[in_proj_key].shape)} implies embed_dim={embed_dim}, but "
            f"arch.obs_encoding_size={arch.obs_encoding_size}."
        )

    num_layers = len(layer_indices)
    if num_layers != arch.mha_num_attention_layers:
        raise ValueError(
            f"edge_arch mismatch: state_dict has {num_layers} 'decoder.sa_decoder.layers.*' "
            f"transformer layers, but arch.mha_num_attention_layers="
            f"{arch.mha_num_attention_layers}."
        )
