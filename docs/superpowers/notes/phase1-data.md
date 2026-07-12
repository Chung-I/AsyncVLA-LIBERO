# Phase 1 data — LIBERO-Spatial training dataset (Task 4.1)

## Data source resolved

- **Dataset**: `openvla/modified_libero_rlds` on the HF Hub (dataset repo, not model repo),
  config directory `libero_spatial_no_noops/1.0.0/`. This is the exact OpenVLA-OFT RLDS
  release referenced by `../openvla-oft/LIBERO.md` ("download the LIBERO datasets ...
  `git clone git@hf.co:datasets/openvla/modified_libero_rlds`") and matches
  `prismatic/vla/datasets/rlds/oxe/configs.py:645`'s `libero_spatial_no_noops` OXE config
  (`image_obs_keys={"primary": "image", "wrist": "wrist_image"}`,
  `state_obs_keys=["EEF_state", "gripper_state"]`).
- **Format**: TFDS/RLDS, one `tf.Example` record per **episode** (not per step) — each
  record's `steps` field is a nested ragged `tf.data.Dataset` of per-timestep dicts.
  `train` split only, 16 shards, `dataset_info.json` reports 432 total episodes
  (27 episodes/shard × 16 shards), ~1.78 GiB uncompressed for the full split.
- **Keys** (from `features.json`, confirmed by reading real records):
  - `episode_metadata.file_path` (text)
  - per-step `steps.observation.image` — agentview RGB, `(256, 256, 3)` uint8, JPEG-encoded
  - per-step `steps.observation.wrist_image` — wrist RGB, `(256, 256, 3)` uint8, JPEG-encoded
  - per-step `steps.observation.state` — `(8,)` float32, "Robot EEF state (6D pose, 2D
    gripper)" — this is the SAME pre-baked 8-dim proprio vector
    (`eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)`) that
    `experiments/robot/openvla_utils.py::get_vla_action` builds live from the sim at eval
    time; matches `LiberoBaseConfig`'s `PROPRIO_DIM = 8`.
  - per-step `steps.observation.joint_state` — `(7,)` float32 (unused here)
  - per-step `steps.action` — `(7,)` float32, raw (un-normalized) EEF delta-action + gripper
  - per-step `steps.language_instruction` — text, identical across all steps of an episode
  - per-step `steps.{is_first,is_last,is_terminal,discount,reward}` — unused here

## Download / smoke-test provenance (what THIS task actually did — do not re-derive)

Full download is ~1.78 GiB and is deferred to the NCHC Nano4 training run (out of scope for
local smoke-testing per the task brief). For local development, downloaded the **minimum**
needed to exercise the real code path:

```bash
# metadata (tiny) + exactly 1 of 16 tfrecord shards (~113 MB), via huggingface_hub
# (destination: data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/, gitignored)
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
dest = "data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
os.makedirs(dest, exist_ok=True)
for fn in [
    "libero_spatial_no_noops/1.0.0/dataset_info.json",
    "libero_spatial_no_noops/1.0.0/features.json",
    "libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016",
]:
    p = hf_hub_download(repo_id="openvla/modified_libero_rlds", repo_type="dataset", filename=fn)
    shutil.copy(p, os.path.join(dest, os.path.basename(fn)))
PY
```

`tensorflow_datasets.builder_from_directory(...)` happily reads this directory even though
only 1/16 shards is present (it globs whatever `*.tfrecord-*` files exist; the
`dataset_info.json`'s `num_examples=432` is metadata, not a hard requirement at read time).
27 real episodes (the full contents of shard 0) are available locally as a result, each
~100-140 steps — plenty to exercise `LiberoSpatialDataset` end-to-end.

**Full-dataset download command for the Nano4 training run** (all 16 shards, all 4 task
suites, per `../openvla-oft/LIBERO.md`):

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## What `LiberoSpatialDataset` actually reads locally right now

- `prismatic/vla/datasets/libero_dataset.py::DEFAULT_DATA_DIR` points at
  `data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/` (gitignored; not committed).
- If that directory has any `*.tfrecord-*` files (true here — the 1 shard above),
  `LiberoSpatialDataset` reads **real** LIBERO-Spatial episodes via
  `tensorflow_datasets.builder_from_directory` (up to `episode_limit` episodes, default 2,
  lazily capped — no full-split iteration).
- If that directory is absent/empty (e.g. a fresh clone before the Nano4-side full
  download), `LiberoSpatialDataset` falls back to a tiny in-memory synthetic fixture
  (`_make_fixture_episodes`: 2 episodes × 12 steps, random images/proprio/actions) with the
  **identical** per-episode key structure, so `__getitem__` exercises the exact same
  processing code path either way. Verified manually (see task-4.1 report) by pointing
  `data_dir` at a nonexistent path — same shapes/keys result.
- The committed test suite (`tests/test_libero_dataset.py`) runs against whichever path is
  available in the environment (real data here; fixture-safe elsewhere) — it makes no
  assumption about which one is active, only about the resulting item contract.

## Normalization used for training-time base inputs

`LiberoSpatialDataset` normalizes raw `action`/`state` (proprio) into the [-1, 1]
BOUNDS_Q99 space the frozen base's `ActionTokenizer` and `proprio_projector` were trained
on, using the checkpoint's own `dataset_statistics.json`
(`moojink/openvla-7b-oft-finetuned-libero-spatial`, key `libero_spatial_no_noops`) — the
same stats file `experiments/robot/openvla_utils.py::normalize_proprio` uses at eval time.
This numpy port (`_bounds_q99_normalize`) reproduces
`prismatic/vla/datasets/rlds/utils/data_utils.py::normalize_action_and_proprio`'s
BOUNDS_Q99 branch exactly, including the per-dimension `mask` (the checkpoint's action
stats mask the gripper dimension — index 6 — as `False`, i.e. left un-normalized/raw;
proprio has no mask, i.e. all 8 dims normalized).

`gt_action_chunk` (fed to the edge adapter for supervision) stores the **normalized**
action chunk — the same space the base's action tokens / action head operate in — not raw
robot-frame actions.
