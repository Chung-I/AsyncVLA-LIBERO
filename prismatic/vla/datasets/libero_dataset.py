"""
libero_dataset.py

Map-style PyTorch `Dataset` for LIBERO-Spatial that produces items usable by BOTH:

  - the **frozen OpenVLA-OFT base** (`experiments/robot/libero/base_config.py`) —
    `input_ids` / `attention_mask` / `pixel_values` (agentview + wrist, channel-stacked)
    / `labels` / `proprio`, with the ground-truth action chunk teacher-forced into the
    token sequence (same convention as `RLDSBatchTransform` below); and
  - the **AsyncVLA edge adapter** — `obs_img_96` / `past_img_96` (current + previous
    agentview frame, 96x96, ImageNet-normalized) and `gt_action_chunk`
    (`[NUM_ACTIONS_CHUNK, ACTION_DIM]`, BOUNDS_Q99-normalized, same space the base's
    `ActionTokenizer` bins into).

Data source
-----------
Prefers the real LIBERO-Spatial RLDS demos: HF dataset `openvla/modified_libero_rlds`,
config `libero_spatial_no_noops` (the OpenVLA-OFT RLDS release; matches the frozen
checkpoint's normalization/training data). Read directly via
`tensorflow_datasets.builder_from_directory` against a local (possibly partial —
as few as 1 of 16 shards) download; see `docs/superpowers/notes/phase1-data.md` for
the exact download command, keys, and provenance.

If that local data isn't present (e.g. fresh clone before the NCHC full-dataset
download), falls back to a tiny in-memory synthetic fixture with the identical key
structure, so this module and its tests never require the ~10GB full download.

Base-input construction mirrors:
  - `experiments/robot/openvla_utils.py::get_vla_action` for image order
    (agentview primary, then wrist) and the exact prompt string; and
  - `prismatic/vla/datasets/datasets.py::RLDSBatchTransform` for action-chunk
    tokenization (`ActionTokenizer`) and `IGNORE_INDEX` label masking.

See the task-4.1 report for the empirical verification (against the real
`moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint) that this construction
makes `get_current_action_mask(...) | get_next_actions_mask(...)` select exactly
`NUM_ACTIONS_CHUNK * ACTION_DIM == 56` positions through the frozen base's forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

# === Checkpoint / dataset identity (must match experiments/robot/libero/base_config.py) ===
PRETRAINED_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial_no_noops"
PROPRIO_DIM = 8

# Default local data location for a (possibly partial) `modified_libero_rlds` download.
# See docs/superpowers/notes/phase1-data.md for the exact download command.
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "modified_libero_rlds" / "libero_spatial_no_noops" / "1.0.0"
)

# === Vision-patch bookkeeping (verified against the real frozen base; see task-4.1 report) ===
# `vla.vision_backbone.get_num_patches()` for the dinosiglip-224px fused ViT (224 / 14 = 16x16 grid).
NUM_PATCHES_PER_IMAGE = 256
# agentview + wrist, matches `LiberoBaseConfig.num_images_in_input`.
NUM_IMAGES_IN_INPUT = 2
# `OpenVLAForActionPrediction._process_proprio_features` appends exactly one proprio token
# after the vision patches (see `prismatic/extern/hf/modeling_prismatic.py`); the checkpoint's
# `LiberoBaseConfig.use_proprio=True`, so this is always included.
PROPRIO_TOKEN_COUNT = 1
NUM_BASE_PATCHES = NUM_PATCHES_PER_IMAGE * NUM_IMAGES_IN_INPUT + PROPRIO_TOKEN_COUNT  # 513

# ImageNet normalization for the 96x96 edge frames — matches `transform` in
# `inference/run_asyncvla.py`, applied to a `TF.resize(TF.to_tensor(img), (96, 96))` frame.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_edge_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _to_edge_frame(img: Image.Image) -> torch.Tensor:
    """Resize a PIL image to 96x96, to_tensor, ImageNet-normalize -> [3, 96, 96]."""
    frame = TF.resize(TF.to_tensor(img.convert("RGB")), (96, 96))
    return _edge_normalize(frame)


def _apply_libero_action_transform(action_chunk_raw: np.ndarray) -> np.ndarray:
    """Numpy port of `prismatic/vla/datasets/rlds/oxe/transforms.py::libero_dataset_transform`
    (the OXE standardization transform registered for `libero_spatial_no_noops` in
    `OXE_STANDARDIZATION_TRANSFORMS`), applied to a raw `[T, ACTION_DIM]` action chunk read
    directly from the RLDS tfrecords.

    That transform only touches the action's gripper dim (index 6): raw LIBERO gripper
    actions are in [-1 (open), 1 (close)]; the transform clips to [0, 1] then inverts
    (`invert_gripper_actions`, i.e. `1 - x`) so the standardized convention is
    +1 = open, 0 = close -- matching the frozen checkpoint's `dataset_statistics.json`
    (dim 6: min=0, max=1). It does NOT touch the other 6 action dims, and its only effect
    on `observation` is renaming/re-slicing `state` into `EEF_state`/`gripper_state`
    (no value changes), which this dataset does not use (proprio here is the raw state
    concat, already verified correct).
    """
    action_chunk = action_chunk_raw.copy()
    action_chunk[:, 6] = 1.0 - np.clip(action_chunk[:, 6], 0.0, 1.0)
    return action_chunk


def _bounds_q99_normalize(x: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Numpy port of the BOUNDS_Q99 branch of
    `prismatic/vla/datasets/rlds/utils/data_utils.py::normalize_action_and_proprio`
    (also matches `experiments/robot/openvla_utils.py::normalize_proprio`).
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
    normalized = np.clip(2 * (x - q01) / (q99 - q01 + 1e-8) - 1, -1.0, 1.0)
    return np.where(mask, normalized, x).astype(np.float32)


def _load_dataset_statistics(unnorm_key: str = UNNORM_KEY) -> Dict[str, Any]:
    """Fetch the checkpoint's small `dataset_statistics.json` (HF-cached) and return the
    `unnorm_key` entry — used to normalize raw actions/proprio into the [-1, 1] BOUNDS_Q99
    space the frozen base + `ActionTokenizer` were trained on.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=PRETRAINED_CHECKPOINT, filename="dataset_statistics.json")
    with open(path, "r") as f:
        stats = json.load(f)
    return stats[unnorm_key]


def load_processor():
    """Loads the checkpoint's HF `processor` (tokenizer + image processor) only —
    lightweight relative to the full 7B model (matches
    `experiments/robot/openvla_utils.py::get_processor`).
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    return AutoProcessor.from_pretrained(PRETRAINED_CHECKPOINT, trust_remote_code=True)


def build_prompt(task_label: str) -> str:
    """Exact prompt format used at both eval (`get_vla_action`) and training time."""
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def build_action_chunk_string(action_tokenizer: ActionTokenizer, action_chunk: np.ndarray) -> str:
    """Tokenizes an `[NUM_ACTIONS_CHUNK, ACTION_DIM]` (already normalized to [-1, 1]) action
    chunk into its concatenated action-token string (current action, then future actions) —
    same order as `RLDSBatchTransform`.
    """
    current_action_string = action_tokenizer(action_chunk[0])
    future_actions_string = "".join(action_tokenizer(action_chunk[1:]))
    return current_action_string + future_actions_string


def build_base_item(
    *,
    processor,
    action_tokenizer: ActionTokenizer,
    task_label: str,
    primary_image: Image.Image,
    wrist_image: Image.Image,
    proprio: np.ndarray,
    action_chunk: np.ndarray,
    predict_stop_token: bool = True,
) -> Dict[str, torch.Tensor]:
    """Builds one training item's base-model fields: `input_ids`, `attention_mask`,
    `pixel_values` (agentview + wrist, channel-stacked), `labels`, `proprio`.

    The ground-truth `action_chunk` (already normalized to [-1, 1]) is teacher-forced
    into `input_ids` as `ActionTokenizer` tokens, immediately after "Out:" + a literal
    space — reproducing the space token (id 29871) that
    `OpenVLAForActionPrediction.predict_action` inserts after "Out:" at inference time
    to match the training distribution. `labels` masks everything except the action
    chunk (+ stop token) with `IGNORE_INDEX`, matching `RLDSBatchTransform`.
    """
    action_chunk_string = build_action_chunk_string(action_tokenizer, action_chunk)
    action_chunk_len = len(action_chunk_string)

    prompt = build_prompt(task_label)
    full_text = prompt + " " + action_chunk_string + "</s>"

    input_ids = torch.tensor(processor.tokenizer(full_text, add_special_tokens=True).input_ids)
    labels = input_ids.clone()
    labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
    if not predict_stop_token:
        labels[-1] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)

    primary_pixel_values = processor.image_processor.apply_transform(primary_image.convert("RGB"))
    wrist_pixel_values = processor.image_processor.apply_transform(wrist_image.convert("RGB"))
    pixel_values = torch.cat([primary_pixel_values, wrist_pixel_values], dim=0)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        proprio=torch.tensor(np.asarray(proprio, dtype=np.float32)),
    )


def _has_tfrecords(data_dir: Path) -> bool:
    return data_dir.is_dir() and any(data_dir.glob("*.tfrecord-*"))


def _load_real_episodes(data_dir: Path, split: str, episode_limit: int) -> List[Dict[str, Any]]:
    """Reads up to `episode_limit` episodes from a local `libero_spatial_no_noops` RLDS
    directory (works even with only some of the 16 shards present locally).
    """
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(data_dir))
    ds = builder.as_dataset(split=split)

    episodes: List[Dict[str, Any]] = []
    for ep in ds:
        images, wrists, states, actions = [], [], [], []
        lang = None
        for step in ep["steps"]:
            images.append(step["observation"]["image"].numpy())
            wrists.append(step["observation"]["wrist_image"].numpy())
            states.append(step["observation"]["state"].numpy())
            actions.append(step["action"].numpy())
            if lang is None:
                lang = step["language_instruction"].numpy().decode()
        episodes.append(dict(image=images, wrist_image=wrists, state=states, action=actions, language_instruction=lang))
        if len(episodes) >= episode_limit:
            break
    return episodes


def _make_fixture_episodes(n_episodes: int = 2, steps_per_episode: int = 12, seed: int = 0) -> List[Dict[str, Any]]:
    """Tiny in-memory fixture with the exact same per-episode key structure as
    `_load_real_episodes`, so `LiberoSpatialDataset.__getitem__` exercises the identical
    code path when the real RLDS download isn't present locally.
    """
    rng = np.random.RandomState(seed)
    tasks = [
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the red mug in the top drawer and place it on the plate",
    ]
    episodes = []
    for i in range(n_episodes):
        images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(steps_per_episode)]
        wrists = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(steps_per_episode)]
        states = [rng.uniform(-1, 1, size=(PROPRIO_DIM,)).astype(np.float32) for _ in range(steps_per_episode)]
        actions = [rng.uniform(-1, 1, size=(ACTION_DIM,)).astype(np.float32) for _ in range(steps_per_episode)]
        episodes.append(
            dict(
                image=images,
                wrist_image=wrists,
                state=states,
                action=actions,
                language_instruction=tasks[i % len(tasks)],
            )
        )
    return episodes


class LiberoSpatialDataset(Dataset):
    """Map-style LIBERO-Spatial dataset; see module docstring for the exact item contract."""

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
        processor=None,
        episode_limit: int = 2,
        predict_stop_token: bool = True,
    ) -> None:
        self.predict_stop_token = predict_stop_token
        self.processor = processor if processor is not None else load_processor()
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        self._action_stats: Optional[Dict[str, Any]] = None
        self._proprio_stats: Optional[Dict[str, Any]] = None
        try:
            stats = _load_dataset_statistics()
            self._action_stats = stats["action"]
            self._proprio_stats = stats["proprio"]
        except Exception:
            # Offline / no HF access: fall back to un-normalized values. Real training runs
            # (NCHC Nano4) always have the checkpoint (and thus these stats) cached locally.
            pass

        resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        if _has_tfrecords(resolved_data_dir):
            self._episodes = _load_real_episodes(resolved_data_dir, split=split, episode_limit=episode_limit)
        else:
            self._episodes = _make_fixture_episodes()

        # Flatten (episode_idx, t) pairs for every start `t` with a full NUM_ACTIONS_CHUNK-length
        # future window — matches the RLDS `chunk_act_obs` convention (no tail padding).
        self._index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            traj_len = len(ep["action"])
            for t in range(max(traj_len - NUM_ACTIONS_CHUNK + 1, 0)):
                self._index.append((ep_idx, t))
                if max_samples is not None and len(self._index) >= max_samples:
                    break
            if max_samples is not None and len(self._index) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        ep = self._episodes[ep_idx]

        primary_now = Image.fromarray(ep["image"][t])
        primary_prev = Image.fromarray(ep["image"][t - 1]) if t > 0 else primary_now
        wrist_now = Image.fromarray(ep["wrist_image"][t])

        proprio_raw = np.asarray(ep["state"][t], dtype=np.float32)
        proprio = _bounds_q99_normalize(proprio_raw, self._proprio_stats) if self._proprio_stats else proprio_raw

        action_chunk_raw = np.stack(ep["action"][t : t + NUM_ACTIONS_CHUNK]).astype(np.float32)
        # Replicate the OXE `libero_dataset_transform` standardization (gripper convention
        # fix) that the frozen checkpoint was trained through, BEFORE normalization/tokenization.
        action_chunk_raw = _apply_libero_action_transform(action_chunk_raw)
        action_chunk = (
            _bounds_q99_normalize(action_chunk_raw, self._action_stats) if self._action_stats else action_chunk_raw
        )

        task_label = ep["language_instruction"]

        item = build_base_item(
            processor=self.processor,
            action_tokenizer=self.action_tokenizer,
            task_label=task_label,
            primary_image=primary_now,
            wrist_image=wrist_now,
            proprio=proprio,
            action_chunk=action_chunk,
            predict_stop_token=self.predict_stop_token,
        )
        item["obs_img_96"] = _to_edge_frame(primary_now)
        item["past_img_96"] = _to_edge_frame(primary_prev)
        item["gt_action_chunk"] = torch.tensor(action_chunk, dtype=torch.float32)
        return item


def collate_libero_batch(instances: Sequence[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pads/stacks a list of `LiberoSpatialDataset` items into a batch."""
    input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=torch.stack([inst["pixel_values"] for inst in instances]),
        proprio=torch.stack([inst["proprio"] for inst in instances]),
        obs_img_96=torch.stack([inst["obs_img_96"] for inst in instances]),
        past_img_96=torch.stack([inst["past_img_96"] for inst in instances]),
        gt_action_chunk=torch.stack([inst["gt_action_chunk"] for inst in instances]),
    )


def make_dummy_base_batch(
    processor,
    task_label: str = "pick up the black bowl and place it on the plate",
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Builds ONE valid-format base batch (real processor + real prompt + a synthetic
    image/proprio/action) for smoke-testing the frozen base forward (used by the
    hidden-state-extraction task that follows this one).

    Returns `(batch_dict, num_patches)` where `num_patches` is the number of tokens the
    base prepends to the text sequence: `NUM_PATCHES_PER_IMAGE * NUM_IMAGES_IN_INPUT +
    PROPRIO_TOKEN_COUNT == 513`. Verified against the real
    `moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint (see the task-4.1
    report): running the base forward on this exact batch and computing
    `get_current_action_mask(labels[:, 1:]) | get_next_actions_mask(labels[:, 1:])`
    selects exactly `NUM_ACTIONS_CHUNK * ACTION_DIM == 56` positions, and
    `hidden_states[-1][:, num_patches:-1]` aligns 1:1 with `labels[:, 1:]`.
    """
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    rng = np.random.RandomState(seed)
    action_chunk = rng.uniform(-1, 1, size=(NUM_ACTIONS_CHUNK, ACTION_DIM)).astype(np.float32)
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)

    synthetic_image = Image.new("RGB", (224, 224), color=(128, 100, 60))

    item = build_base_item(
        processor=processor,
        action_tokenizer=action_tokenizer,
        task_label=task_label,
        primary_image=synthetic_image,
        wrist_image=synthetic_image,
        proprio=proprio,
        action_chunk=action_chunk,
        predict_stop_token=True,
    )

    batch = {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "pixel_values": item["pixel_values"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
        "proprio": item["proprio"].unsqueeze(0),
    }
    return batch, NUM_BASE_PATCHES
