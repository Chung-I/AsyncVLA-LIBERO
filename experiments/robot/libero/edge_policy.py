r"""EdgePolicy: base -> projector -> edge adapter action-chunk policy for LIBERO-Spatial.

Phase-1 (SYNCHRONOUS) rollout policy: the base and edge both run every control step
(mirrors `inference/run_asyncvla.py::run_forward_pass`, retargeted to manipulation the same
way `vla-scripts/train_asyncvla_libero.py::run_forward_pass` does for training). The
predicted 8-step chunk is then executed open-loop at the SAME cadence `--mode stock` uses
(`run_libero_eval.py`'s `action_queue`) -- this is NOT the two-rate async loop (that's
Phase 2).

Chain per query (`EdgePolicy.act`):
  1. Base-batch construction mirrors `experiments.robot.openvla_utils.get_vla_action`
     (image order: agentview primary, then wrist; exact prompt string; proprio
     normalization) via `prismatic.vla.datasets.libero_dataset.build_base_item`, fed a
     dummy (zero) action chunk as the teacher-forced placeholder in `input_ids`. The
     placeholder's VALUE doesn't matter for the extracted hidden states:
     `OpenVLAForActionPrediction.forward` zeroes the action-token embeddings before the LM
     forward (see `prismatic/extern/hf/modeling_prismatic.py`, the `else: # Replace ...
     with zeros` branch) -- only the placeholder's fixed *positions*
     (`NUM_ACTIONS_CHUNK * ACTION_DIM == 56` tokens) matter, which `build_base_item`
     reproduces exactly regardless of the action values tokenized into them.
  2. `extract_actions_hidden_states(vla, batch, proprio_projector, num_patches=513)` ->
     `[B, 56, 4096]` bf16 hidden states (the `vla_feature` source).
  3. `proj.predict_action(hidden.float(), taskid=zeros)` -> `vla_feature` `[B, 8, 512]`.
  4. `edge(obs_img_96, past_img_96, vla_feature)` -> `[B, 8, 7]` predicted action chunk, in
     the SAME standardized/normalized target space the base's own L1 action head predicts
     in (BOUNDS_Q99, gripper dim masked/pass-through) -- see the module docstring of
     `prismatic/vla/datasets/libero_dataset.py`.
  5. Unnormalize via `vla._unnormalize_actions(..., unnorm_key)` -- the exact method
     `OpenVLAForActionPrediction.predict_action` calls for `--mode stock`
     (`prismatic/extern/hf/modeling_prismatic.py` ~line 1578) -- so both modes hand
     `run_libero_eval.py::process_action` (gripper normalize+invert) the same target space
     before stepping the env.

Edge input frames (`obs_img_96`/`past_img_96`): current RAW agentview frame (pre-resize,
pre-center-crop -- `get_libero_image(obs)`'s raw rotated output), resized to 96x96 and
ImageNet-normalized via `prismatic.vla.datasets.libero_dataset._to_edge_frame` -- the exact
transform `LiberoSpatialDataset` uses to build `obs_img_96`/`past_img_96` at training time.
`past_img_96` is the 96px frame captured the last time the base was refreshed (see
`_should_refresh`, `EdgeEvalConfig.base_cadence`): with `base_cadence=1` (default) the base
refreshes every call, so `past == obs` on every call, matching Phase-1 behavior; with
`base_cadence > 1`, `past_img_96` holds the frame from the last refresh while `obs_img_96`
is always the current frame, giving the edge a stale/fresh frame pair mirroring the
stale/fresh `vla_feature`. At the first call of an episode (cache empty after `reset()`),
`past == obs` regardless of `base_cadence`, matching `LiberoSpatialDataset.__getitem__`'s
`t == 0` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
from experiments.robot.libero.base_features import extract_actions_hidden_states
from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.robot.openvla_utils import DEVICE, normalize_proprio, prepare_images_for_vla
from prismatic.models.small_head import Edge_adapter_manip, Proj_Actiontokens
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.libero_dataset import NUM_BASE_PATCHES, _to_edge_frame, build_base_item

# Must match `vla-scripts/train_asyncvla_libero.py::EDGE_OBS_ENCODING_SIZE` -- the
# edge/proj checkpoints being loaded here were trained with this dim.
EDGE_OBS_ENCODING_SIZE = 512


@dataclass
class EdgeEvalConfig(LiberoBaseConfig):
    """Extends `LiberoBaseConfig` with the extra fields `EdgePolicy` needs: `center_crop`
    (read by `experiments.robot.openvla_utils.prepare_images_for_vla`, same as
    `run_libero_eval.py::StockEvalConfig`) and `unnorm_key` (resolved by `EdgePolicy.__init__`
    via `resolve_unnorm_key`, then read by `act()` for action unnormalization)."""

    center_crop: bool = True
    unnorm_key: str = ""
    base_cadence: int = 1
    """Number of `EdgePolicy.act()` calls between frozen-base recomputes. `1` (default)
    recomputes `vla_feature` every call, reproducing Phase-1 `EdgePolicy` behavior exactly.
    `N > 1` recomputes on calls `0, N, 2N, ...` and holds the cached `vla_feature` (and the
    base's 96px agentview frame captured at that recompute) stale for the edge on the calls
    in between -- see `_should_refresh`."""


def _should_refresh(step: int, cadence: int) -> bool:
    """Returns whether the frozen base's `vla_feature` should be recomputed at `step`
    (0-indexed act() call count), given a `base_cadence` of `cadence` act() calls between
    base refreshes. `cadence == 1` refreshes every call (Phase-1 behavior); `cadence == N`
    refreshes on calls `0, N, 2N, ...`, holding the cached feature/frame stale in between."""
    return step % cadence == 0


def resolve_unnorm_key(vla, task_suite_name: str) -> str:
    """Resolves `unnorm_key` for `vla.norm_stats`: `task_suite_name`, falling back to
    `f"{task_suite_name}_no_noops"` if the plain name isn't present. Identical logic to
    `--mode stock`'s resolution in `run_libero_eval.py::main` -- factored here so both
    modes (`run_libero_eval.py` imports this for `--mode stock` too) always unnormalize
    against the same checkpoint statistics via the same code path.
    """
    unnorm_key = task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, (
        f"Action un-norm key {unnorm_key!r} not found in VLA norm_stats! Available: {list(vla.norm_stats.keys())}"
    )
    return unnorm_key


def _strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strips a `module.` prefix left over from a `DistributedDataParallel`-wrapped
    checkpoint. `vla-scripts/train_asyncvla_libero.py::save_training_checkpoint` already
    unwraps DDP before saving, so this is normally a no-op; kept defensively for
    checkpoints saved by other means."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    return state_dict


class EdgePolicy:
    """Synchronous (Phase-1) base -> projector -> edge action-chunk policy.

    `act(obs, task_description)` returns a list of `NUM_ACTIONS_CHUNK` unnormalized
    (env-ready, PRE `run_libero_eval.py::process_action` gripper remap) `[ACTION_DIM]`
    arrays -- the same return contract `experiments.robot.openvla_utils.get_vla_action`
    has for `--mode stock`, so the rollout loop (action queue, `process_action`,
    open-loop cadence) is unchanged between modes.
    """

    def __init__(
        self,
        cfg: EdgeEvalConfig,
        task_suite_name: str,
        edge_ckpt: str,
        proj_ckpt: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device if device is not None else DEVICE

        self.vla, self.processor, self.proprio_projector = build_frozen_base(cfg)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        self.unnorm_key = resolve_unnorm_key(self.vla, task_suite_name)
        self.cfg.unnorm_key = self.unnorm_key

        self.proj = Proj_Actiontokens(
            input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=EDGE_OBS_ENCODING_SIZE
        )
        proj_state = _strip_ddp_prefix(torch.load(proj_ckpt, map_location="cpu"))
        self.proj.load_state_dict(proj_state)
        self.proj.requires_grad_(False)
        self.proj.eval()
        self.proj.to(device=self.device, dtype=torch.float32)

        self.edge = Edge_adapter_manip(obs_encoding_size=EDGE_OBS_ENCODING_SIZE)
        edge_state = _strip_ddp_prefix(torch.load(edge_ckpt, map_location="cpu"))
        self.edge.load_state_dict(edge_state)
        self.edge.requires_grad_(False)
        self.edge.eval()
        self.edge.to(device=self.device, dtype=torch.float32)

        self.base_cadence = cfg.base_cadence
        self._step = 0
        self._cached_feature: Optional[torch.Tensor] = None
        self._cached_base_frame96: Optional[torch.Tensor] = None
        # Diagnostic counter (not used by `act`'s control flow): counts how many times the
        # `_should_refresh(...) or cache-empty` guard in `act()` actually recomputed the
        # frozen base, for mechanically verifying cadence in an eval harness (over an
        # S-step episode, expect `ceil(S / base_cadence)` recomputes vs. S edge calls).
        self._base_recompute_count = 0

    def reset(self) -> None:
        """Resets the step counter and clears the base-feature/frame cache. Call at the
        start of each episode so the first `act()` call recomputes the base (cache is
        empty) and `past_img_96 == obs_img_96` at that first call."""
        self._step = 0
        self._cached_feature = None
        self._cached_base_frame96 = None
        self._base_recompute_count = 0

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], task_description: str) -> List[np.ndarray]:
        """Runs base -> projector -> edge on the current raw LIBERO env `obs` and returns
        the predicted `[NUM_ACTIONS_CHUNK, ACTION_DIM]` action chunk as a list of
        `NUM_ACTIONS_CHUNK` unnormalized `[ACTION_DIM]` arrays.

        Async cache (Phase-2): the frozen base's `vla_feature` (and the 96px agentview
        frame captured alongside it) is recomputed only every `self.base_cadence` calls
        (`_should_refresh`) or when the cache is empty (first call after `reset()`); the
        edge runs every call on the CURRENT 96px frame plus the cached (possibly stale)
        frame/feature pair. `base_cadence=1` refreshes on every call, so the cached frame
        is always the just-captured current frame -- identical to Phase-1 `EdgePolicy.act`.
        """
        full_image = get_libero_image(obs)  # raw 256x256 uint8 agentview, rotated 180
        wrist_image = get_libero_wrist_image(obs)

        current_pil = Image.fromarray(full_image)
        obs_img_96 = _to_edge_frame(current_pil).unsqueeze(0).to(self.device)

        if _should_refresh(self._step, self.base_cadence) or self._cached_feature is None:
            proprio_raw = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            proprio_norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio_raw, proprio_norm_stats)

            # Base-input image prep: identical helper `get_vla_action` uses (resize +
            # optional center-crop), so the frozen base sees the same image distribution
            # in both modes.
            primary_pil, wrist_pil = prepare_images_for_vla([full_image, wrist_image], self.cfg)

            dummy_action_chunk = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
            item = build_base_item(
                processor=self.processor,
                action_tokenizer=self.action_tokenizer,
                task_label=task_description,
                primary_image=primary_pil,
                wrist_image=wrist_pil,
                proprio=proprio,
                action_chunk=dummy_action_chunk,
                predict_stop_token=True,
            )
            batch = {k: v.unsqueeze(0) for k, v in item.items()}

            hidden = extract_actions_hidden_states(
                self.vla, batch, self.proprio_projector, num_patches=NUM_BASE_PATCHES, device=self.device,
            )  # [1, 56, 4096] bf16

            taskid = torch.zeros(1, device=self.device)
            self._cached_feature = self.proj.predict_action(hidden.to(torch.float32), taskid)  # [1, 8, 512] fp32
            self._cached_base_frame96 = obs_img_96
            self._base_recompute_count += 1

        pred_chunk = self.edge(obs_img_96, self._cached_base_frame96, self._cached_feature)  # [1, 8, 7] fp32, normalized space

        normalized_actions = pred_chunk[0].detach().cpu().numpy().astype(np.float32)
        actions = self.vla._unnormalize_actions(normalized_actions, self.cfg.unnorm_key)  # [8, 7]

        self._step += 1
        return [actions[i] for i in range(actions.shape[0])]
