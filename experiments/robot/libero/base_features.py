r"""Extract action-token hidden states (the `vla_feature` source for the edge adapter)
from the frozen OpenVLA-OFT LIBERO base.

Mirrors `inference/run_asyncvla.py::run_forward_pass` (lines ~521-553), adapted to the
LIBERO base's actual forward signature:

  - `experiments/robot/libero/base_config.py::build_frozen_base` loads this checkpoint
    with `use_film=False` (it ships no FiLM-wrapped vision-backbone weights) and
    `use_proprio=True` (proprio_dim=8).
  - The base forward used here is `OpenVLAForActionPrediction.forward` (the plain
    vanilla-VLM signature in `prismatic/extern/hf/modeling_prismatic.py` ~line 501:
    `input_ids, attention_mask, pixel_values, labels, output_hidden_states, proprio,
    proprio_projector, noisy_actions, noisy_action_projector,
    diffusion_timestep_embeddings, use_film`), NOT the `modality_id`/pose-goal-navigation
    forward variant (~line 1017) that `run_asyncvla.py` calls for the DROID/satellite
    model -- LIBERO has no `modality_id`, `goal_pose`, or diffusion action head, so those
    kwargs are omitted entirely here (not passed as `None`).
  - `num_patches` is `NUM_PATCHES_PER_IMAGE * NUM_IMAGES_IN_INPUT + PROPRIO_TOKEN_COUNT
    == 513` for this checkpoint (256 vision patches x 2 images [agentview + wrist] + 1
    proprio token) -- see `prismatic/vla/datasets/libero_dataset.py::NUM_BASE_PATCHES`,
    the source of truth; pass it in rather than hardcoding, but callers should use that
    constant.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@torch.no_grad()
def extract_actions_hidden_states(
    vla,
    batch: Dict[str, torch.Tensor],
    proprio_projector,
    num_patches: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run the frozen LIBERO base and slice out the action-token hidden states.

    Args:
        vla: frozen `OpenVLAForActionPrediction` (from `build_frozen_base`).
        batch: dict with `input_ids`, `attention_mask`, `pixel_values`, `labels`, and
            (when `proprio_projector` is not None) `proprio` -- e.g. from
            `make_dummy_base_batch` or `collate_libero_batch`.
        proprio_projector: frozen proprio projector from `build_frozen_base`, or `None`.
        num_patches: number of vision-patch(+proprio) tokens prepended to the text
            sequence -- `prismatic.vla.datasets.libero_dataset.NUM_BASE_PATCHES` (513)
            for this checkpoint.
        device: device to run the forward pass on; defaults to `vla`'s device.

    Returns:
        `Tensor[B, NUM_ACTIONS_CHUNK * ACTION_DIM, D]` (bfloat16) -- the base's last-layer
        hidden states at the current + future action-token positions. `D == vla.llm_dim`
        (4096 for this checkpoint). This is the `vla_feature` the edge adapter distills.
    """
    if device is None:
        device = next(vla.parameters()).device

    proprio = batch.get("proprio")
    output = vla(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
        labels=batch["labels"].to(device),
        output_hidden_states=True,
        proprio=proprio.to(torch.bfloat16).to(device) if proprio is not None else None,
        proprio_projector=proprio_projector,
        use_film=False,
    )

    last_hidden = output.hidden_states[-1]  # (B, seq_len, D)
    # Text portion of prompt + response, after the vision(+proprio) patch tokens.
    text_hidden = last_hidden[:, num_patches:-1]

    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return actions_hidden_states
