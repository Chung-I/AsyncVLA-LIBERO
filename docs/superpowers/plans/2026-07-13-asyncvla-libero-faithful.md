# AsyncVLA-LIBERO **Faithful** Phase-2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce AsyncVLA's Phase 2 as faithfully as the embodiment allows — original `Edge_adapter` at the original capacity, `k_max=3`, their exact 3-term weighted MSE loss, and their random-crop augmentation — then produce the faithful SR-vs-cadence curve (N∈{1,2,3,4,6,8}) against the validated 0.982 stock topline.

**Architecture:** This worktree **is a git worktree of the AsyncVLA repo**, so the original code is already present unmodified. The method is **minimal-delta**: call the original classes directly (`Edge_adapter`, `Proj_Actiontokens`, `MultiLayerDecoder_trans`), read the original hyperparameters from the original files (`config_nav/dataset_config.yaml`), and change as little as possible. Only four behaviors change vs our current LIBERO stack (capacity, delay range, loss, augmentation); the async eval harness is already faithful and is reused **unchanged**.

**Tech Stack:** PyTorch, HuggingFace Transformers (OpenVLA-OFT fork), EfficientNet-PyTorch, LIBERO + robosuite, RLDS/TFDS, Weights & Biases, Slurm on NCHC Nano4 (H200).

## Global Constraints

- Work ONLY in `/home/chungyili/Codes/AsyncVLA-libero-faithful` (branch `asyncvla-libero-faithful`). Do NOT touch `/home/chungyili/Codes/async_pi0` or the sibling `AsyncVLA-libero` worktree (it has cluster jobs running).
- **Minimal-delta rule:** where the original has code, CALL IT. Do not reimplement or duplicate. Every deviation must already be listed in the spec's §7 ledger — if you find yourself needing a new one, STOP and report it.
- Interpreter: `.venv/bin/python` (no `python` on PATH). The venv is shared via the repo checkout; if `.venv` is absent in this worktree, symlink it: `ln -s ../AsyncVLA-libero/.venv .venv`.
- **Only ONE GPU model-loading job at a time.** Before any GPU/sim command: `pgrep -af run_libero_eval` and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` must be clear. Run GPU tests as a single blocking foreground command; never background or duplicate.
- Faithful values (from the original, do not invent): edge **1024 / 4 heads / 4 layers / ff×4** (`config_nav/dataset_config.yaml`); **`k_max=3`** (`prismatic/vla/datasets/lelan_dataset.py:304`); loss weights **`0.5` / `0.5×15` / `0.1`** (`vla-scripts/train_asyncvla.py:559`); crop **`v_random=0.2`, `h_random=0.1`** (`lelan_dataset.py:113-114,363-371`); AdamW **`lr=1e-4`**.
- Constants: `ACTION_DIM=7` (dims 0–5 = EEF deltas, dim 6 = gripper), `NUM_ACTIONS_CHUNK=8`, `num_patches=513`, `use_film=False`, base `llm_dim=4096`.
- Cadence sweep: **N ∈ {1,2,3,4,6,8}**; eval protocol 10 tasks × 50 trials = **500** per config.
- Cluster: NCHC Nano4 (H200) per `docs/superpowers/notes/nano4-handoff.md`; always `export HF_HOME=/work/roboleon1295/.cache/huggingface MUJOCO_GL=egl TMPDIR=/work/roboleon1295/tmp/job_$SLURM_JOB_ID` (mkdir it) for sim jobs.
- Read eval SR from the run **log**, not the exit code (sim jobs segfault at teardown *after* results are logged).
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01TNfYx4NdAAHHGzyKWqLtn6`

---

## Task 1: Run the ORIGINAL `Edge_adapter`, at the original capacity

Retire our `Edge_adapter_manip` duplicate and call the original class. The original head width becomes a constructor parameter: `nn.Linear(64, 8 * 4)` → `nn.Linear(64, NUM_ACTIONS_CHUNK * action_dim)`, with `action_dim: int = 4` (the original nav width) as the default, so every original caller — which never passes `action_dim` — stays byte-identical (`NUM_ACTIONS_CHUNK * 4 == 8 * 4 == 32`).

**CORRECTION (post-implementation):** an earlier draft of this task described hardcoding `NUM_ACTIONS_CHUNK * ACTION_DIM` directly (no new parameter) as a "semantic no-op for navigation" on the premise that the nav constants are chunk=8 × `ACTION_DIM`=4. That premise is false: `prismatic/vla/constants.py` defines `ACTION_DIM = 7` (the LIBERO value) — there is no `ACTION_DIM=4` anywhere in this codebase. Hardcoding `ACTION_DIM` into the original class would have built a 56-wide head for the original nav callers too, breaking `train_asyncvla.py`'s loss (4-D `daction_ref` shape mismatch) and `run_asyncvla.py`'s checkpoint load (`[32,64]` vs `[56,64]`). The `action_dim` parameter (default 4) is the actual fix; the LIBERO path passes `action_dim=ACTION_DIM` (=7) explicitly via `build_edge_and_proj`. The reshape at the end of `forward` already uses `(batch_size, NUM_ACTIONS_CHUNK, -1)` and adapts for free either way.

**Files:**
- Modify: `prismatic/models/small_head.py` (line **51** inside `class Edge_adapter`; delete `class Edge_adapter_manip` at line **246**)
- Modify: `experiments/robot/libero/edge_arch.py` (import + build + type hints + docstrings)
- Modify: `experiments/robot/libero/latency_bench.py` (docstring/`--help` text only)
- Test: `tests/test_faithful_edge.py`

**Interfaces:**
- Consumes: original `prismatic.models.small_head.Edge_adapter(obs_encoding_size, mha_num_attention_heads, mha_num_attention_layers, mha_ff_dim_factor)`; `Proj_Actiontokens`; `NUM_ACTIONS_CHUNK`, `ACTION_DIM` (already imported in `small_head.py`).
- Produces: `EdgeArch.from_config_nav() -> EdgeArch` (reads `config_nav/dataset_config.yaml`, returns 1024/4/4/4). `build_edge_and_proj(llm_dim, device, arch)` now returns `Tuple[Edge_adapter, Proj_Actiontokens]`. `EdgeArch` field names are unchanged (`obs_encoding_size`, `mha_num_attention_heads`, `mha_num_attention_layers`, `mha_ff_dim_factor`), as are `save_edge_arch(run_dir, arch)` / `load_edge_arch(ckpt_path) -> EdgeArch`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faithful_edge.py
import torch
from prismatic.models.small_head import Edge_adapter
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj


def test_original_edge_adapter_outputs_libero_chunk():
    """The ORIGINAL Edge_adapter, at the ORIGINAL capacity, emits an 8x7 LIBERO chunk."""
    B, D = 2, 1024
    edge = Edge_adapter(
        obs_encoding_size=D, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4
    )
    out = edge(torch.randn(B, 3, 96, 96), torch.randn(B, 3, 96, 96), torch.randn(B, NUM_ACTIONS_CHUNK, D))
    assert out.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM) == (B, 8, 7), out.shape


def test_output_head_width_is_parameterized_not_hardcoded():
    """action_dim is now a constructor kwarg: Linear(64, 8*4) -> Linear(64, NUM_ACTIONS_CHUNK*action_dim).
    Default action_dim=4 (the ORIGINAL nav width) reproduces the original literal `8 * 4 == 32`
    for every original caller. Passing action_dim=ACTION_DIM (=7, LIBERO) gives 56 -- NOT a
    no-op; ACTION_DIM==7 everywhere in this codebase, there is no ACTION_DIM==4."""
    edge_original_default = Edge_adapter(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4)
    assert edge_original_default.action_predictor[-1].out_features == 8 * 4 == 32

    edge_libero = Edge_adapter(
        obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, action_dim=ACTION_DIM
    )
    assert edge_libero.action_predictor[-1].out_features == NUM_ACTIONS_CHUNK * ACTION_DIM == 56


def test_edge_arch_from_config_nav_is_the_original_capacity():
    """Capacity comes from the ORIGINAL config file, not from our class defaults."""
    arch = EdgeArch.from_config_nav()
    assert (arch.obs_encoding_size, arch.mha_num_attention_heads,
            arch.mha_num_attention_layers, arch.mha_ff_dim_factor) == (1024, 4, 4, 4)


def test_build_edge_and_proj_at_faithful_capacity():
    edge, proj = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=EdgeArch.from_config_nav())
    assert isinstance(edge, Edge_adapter)
    assert edge.obs_encoding_size == 1024
    # projector token width MUST equal the edge token width (they are concatenated)
    hidden = torch.randn(2, NUM_ACTIONS_CHUNK * ACTION_DIM, 4096)
    feat = proj.predict_action(hidden, torch.zeros(2))
    assert feat.shape == (2, NUM_ACTIONS_CHUNK, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_faithful_edge.py -v`
Expected: FAIL — `Edge_adapter` still outputs `8*4=32` (shape `(B,8,4)`), and `EdgeArch.from_config_nav` does not exist.

- [ ] **Step 3: Parameterize the head width + point `edge_arch` at the original class**

In `prismatic/models/small_head.py`, add `action_dim: int = 4` to `Edge_adapter.__init__`'s
signature, and change the `action_predictor` output layer (line **51**):
```python
            nn.Linear(64, NUM_ACTIONS_CHUNK * action_dim),
```
(replacing `nn.Linear(64, 8 * 4),`). `NUM_ACTIONS_CHUNK` is already imported at the top of the
file; `ACTION_DIM` is NOT used here (it would be wrong for the original nav callers — see the
correction above). Change nothing else in the class.

Then **delete `class Edge_adapter_manip`** (starts line **246**) entirely — it was only ever `Edge_adapter` with that output parametrized.

In `experiments/robot/libero/edge_arch.py`:
- `from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens` (drop `Edge_adapter_manip`).
- In `build_edge_and_proj`, construct the edge with **all four** arch fields (the current code passes only `obs_encoding_size`, which silently left heads/layers at the class defaults — that was the original bug) **plus `action_dim=ACTION_DIM`** (the LIBERO width, imported from `prismatic.vla.constants`) so the head is 56-wide, not the class default 32-wide:
```python
    edge = Edge_adapter(
        obs_encoding_size=arch.obs_encoding_size,
        mha_num_attention_heads=arch.mha_num_attention_heads,
        mha_num_attention_layers=arch.mha_num_attention_layers,
        mha_ff_dim_factor=arch.mha_ff_dim_factor,
        action_dim=ACTION_DIM,
    )
```
- Update the return type to `Tuple[Edge_adapter, Proj_Actiontokens]` and the docstrings that name `Edge_adapter_manip`.
- Add the classmethod that reads the ORIGINAL config file:
```python
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
```
In `experiments/robot/libero/latency_bench.py`, update the docstring and the four `--help` strings that say `Edge_adapter_manip` to say `Edge_adapter`. No logic change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_faithful_edge.py tests/test_edge_arch.py -v`
Expected: PASS (new faithful-edge tests + the existing `EdgeArch` save/load round-trip tests still green).

- [ ] **Step 5: Commit**

```bash
git add prismatic/models/small_head.py experiments/robot/libero/edge_arch.py \
        experiments/robot/libero/latency_bench.py tests/test_faithful_edge.py
git commit -m "faithful: run the original Edge_adapter at the original 1024/4/4 capacity"
```

---

## Task 2: Their exact 3-term weighted MSE loss

Replace our single L1 with the original's weighted multi-term MSE (`vla-scripts/train_asyncvla.py:552,557,559`), adapted only where the embodiment forces it: their `delta_to_pose` is SE(2) body-frame composition, so the manipulation analog is a **cumsum** over the 6 EEF dims; their LeLaN `obj_pose` term is **dropped**.

**Files:**
- Modify: `vla-scripts/train_asyncvla_libero.py` (add `faithful_chunk_loss`; use it at line **175** in `run_forward_pass`; update the metrics key at line **275** and the module docstring at line **15**)
- Test: `tests/test_faithful_loss.py`

**Interfaces:**
- Consumes: `ACTION_DIM` (=7; dims 0–5 EEF deltas, dim 6 gripper), `torch.nn.functional as F`.
- Produces: `faithful_chunk_loss(pred: Tensor[B,8,7], gt: Tensor[B,8,7]) -> Tuple[Tensor, Dict[str, float]]`. The metrics dict keys are `"loss"`, `"mse_delta"`, `"mse_traj"`, `"mse_smooth"`. **The old `"l1_loss"` key is gone** — every consumer must use `"loss"` (only `run_forward_pass` and the wandb logging at line 275 consume it).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faithful_loss.py
import pytest
import torch
from vla_scripts.train_asyncvla_libero import faithful_chunk_loss


def test_loss_is_their_exact_weighted_sum():
    """Weights are the original's: 0.5*15 (delta), 0.5 (traj), 0.1 (smoothness)."""
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    loss, m = faithful_chunk_loss(pred, gt)
    expected = 0.5 * 15.0 * m["mse_delta"] + 0.5 * m["mse_traj"] + 0.1 * m["mse_smooth"]
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_gripper_excluded_from_traj_and_smoothness_terms():
    """The gripper (dim 6) is an ABSOLUTE command -- it must not be integrated."""
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    _, m0 = faithful_chunk_loss(pred, gt)
    pred2 = pred.clone()
    pred2[..., 6] += 5.0                      # perturb ONLY the gripper dim
    _, m1 = faithful_chunk_loss(pred2, gt)
    assert m1["mse_delta"] != pytest.approx(m0["mse_delta"])      # delta term DOES see it
    assert m1["mse_traj"] == pytest.approx(m0["mse_traj"])        # traj term must NOT
    assert m1["mse_smooth"] == pytest.approx(m0["mse_smooth"])    # smoothness must NOT


def test_smoothness_and_traj_vanish_for_zero_eef_deltas():
    """Zero EEF deltas -> zero integrated trajectory -> sm_ref == traj -> both terms 0."""
    pred, gt = torch.zeros(2, 8, 7), torch.zeros(2, 8, 7)
    pred[..., 6] = 1.0                        # gripper nonzero, EEF all zero
    _, m = faithful_chunk_loss(pred, gt)
    assert m["mse_traj"] == pytest.approx(0.0)
    assert m["mse_smooth"] == pytest.approx(0.0)


def test_loss_is_differentiable():
    pred = torch.randn(2, 8, 7, requires_grad=True)
    loss, _ = faithful_chunk_loss(pred, torch.randn(2, 8, 7))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_faithful_loss.py -v`
Expected: FAIL — `ImportError: cannot import name 'faithful_chunk_loss'`.

- [ ] **Step 3: Implement the loss and use it**

In `vla-scripts/train_asyncvla_libero.py`, add near the top (after the imports; `ACTION_DIM` must be imported from `prismatic.vla.constants` if not already):
```python
# The gripper is the LAST action dim and is an ABSOLUTE command -- it does not integrate.
NUM_EEF_DIMS = ACTION_DIM - 1  # 6


def faithful_chunk_loss(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """AsyncVLA's 3-term weighted MSE, mirroring `vla-scripts/train_asyncvla.py:552,557,559`.

    The original edge emits DELTAS and `delta_to_pose` integrates them into a waypoint
    trajectory; the loss penalizes the deltas (weight 0.5*15), the integrated trajectory
    (0.5) and the trajectory's smoothness (0.1). Their `delta_to_pose` is SE(2) body-frame
    composition (nav-only); LIBERO's OSC deltas are world-frame, so `cumsum` is the analog.
    Their 4th term (0.1 * MSE on the LeLaN object pose) has no LIBERO analog and is dropped.
    """
    pred_traj = torch.cumsum(pred[..., :NUM_EEF_DIMS], dim=1)
    gt_traj = torch.cumsum(gt[..., :NUM_EEF_DIMS], dim=1)
    # sm_ref mirrors `cat(action_orig, predicted_actions[:, 0:-1])`: the previous predicted
    # waypoint, with the chunk's first entry compared against the origin (the current EEF
    # pose is the origin of this relative frame). NOT detached -- matching the original.
    sm_ref = torch.cat([torch.zeros_like(pred_traj[:, :1]), pred_traj[:, :-1]], dim=1)

    mse_delta = F.mse_loss(pred, gt)
    mse_traj = F.mse_loss(pred_traj, gt_traj)
    mse_smooth = F.mse_loss(pred_traj, sm_ref)

    loss = 0.5 * 15.0 * mse_delta + 0.5 * mse_traj + 0.1 * mse_smooth
    metrics = {
        "loss": loss.item(),
        "mse_delta": mse_delta.item(),
        "mse_traj": mse_traj.item(),
        "mse_smooth": mse_smooth.item(),
    }
    return loss, metrics
```
In `run_forward_pass`, replace line **175**'s `loss = F.l1_loss(pred_chunk, gt_action_chunk)` and the `metrics = {"l1_loss": loss.item()}` line with:
```python
    loss, metrics = faithful_chunk_loss(pred_chunk, gt_action_chunk)
    return loss, metrics
```
Update the wandb logging (line **275**) from `recent_l1.append(metrics["l1_loss"])` to `recent_l1.append(metrics["loss"])`, and log the three components too — change the `wandb.log({"train/l1_loss": ...})` call to:
```python
                    wandb.log({
                        "train/loss": sum(recent_l1) / len(recent_l1),
                        "train/mse_delta": metrics["mse_delta"],
                        "train/mse_traj": metrics["mse_traj"],
                        "train/mse_smooth": metrics["mse_smooth"],
                    }, step=step)
```
Update the module docstring line **15** to describe the 3-term loss instead of `F.l1_loss`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_faithful_loss.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add vla-scripts/train_asyncvla_libero.py tests/test_faithful_loss.py
git commit -m "faithful: AsyncVLA's 3-term weighted MSE loss (0.5*15 delta / 0.5 traj / 0.1 smooth)"
```

---

## Task 3: Their random-crop augmentation (one shared box per sample)

Port the original's random crop (`lelan_dataset.py:113-114,363-371`) into the LIBERO dataset. **One box per sample**, shared across all four images — the base's agentview, the base's wrist, and both 96px edge frames — exactly as the original shares one `PILbox` across its base/current/goal frames. Train-only; **no horizontal flip** (their flip mirrors the actions, which is unsafe for 6-DoF + gripper).

**Files:**
- Modify: `prismatic/vla/datasets/libero_dataset.py` (module constants; `__init__`; `__getitem__` lines **377–390**)
- Test: `tests/test_faithful_crop.py`

**Interfaces:**
- Consumes: `LiberoSpatialDataset.__init__(..., delay_aware, k_max, rng_seed)`; the existing `_to_base_image(raw_np) -> Image`, `_to_edge_frame(Image) -> Tensor`.
- Produces: `LiberoSpatialDataset(..., image_aug: bool = False)`. Module-level `V_RANDOM = 0.2`, `H_RANDOM = 0.1`; `_sample_crop_offsets(h, w, rng) -> Tuple[int, int]` returning `(voffset, hoffset)`; `_apply_crop(raw: np.ndarray, voffset: int, hoffset: int) -> np.ndarray`. When `image_aug=True`, `__getitem__` records the sample's single box on `self._last_crop_offsets` (for tests).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faithful_crop.py
import numpy as np
import pytest
import torch
from prismatic.vla.datasets.libero_dataset import (
    DEFAULT_DATA_DIR, H_RANDOM, V_RANDOM, LiberoSpatialDataset,
    _apply_crop, _has_tfrecords, _sample_crop_offsets,
)


def test_crop_offsets_match_the_original_fractions():
    """Original: voffset = int(224 * v_random * rand()), v_random=0.2, h_random=0.1."""
    assert (V_RANDOM, H_RANDOM) == (0.2, 0.1)
    rng = np.random.RandomState(0)
    for _ in range(200):
        v, h = _sample_crop_offsets(256, 256, rng)
        assert 0 <= v < 0.2 * 256
        assert 0 <= h < 0.1 * 256


def test_apply_crop_is_symmetric():
    raw = np.zeros((256, 256, 3), dtype=np.uint8)
    out = _apply_crop(raw, voffset=10, hoffset=5)
    assert out.shape == (256 - 2 * 10, 256 - 2 * 5, 3)


def test_aug_uses_ONE_shared_box_and_is_deterministic_under_seed():
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    a = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    b = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    ia, ib = a[0], b[0]
    # One box per sample, recorded for inspection -- the SAME box drove all four images.
    assert a._last_crop_offsets == b._last_crop_offsets
    assert torch.allclose(ia["obs_img_96"], ib["obs_img_96"])       # deterministic under seed
    assert torch.allclose(ia["pixel_values"], ib["pixel_values"])


def test_aug_off_by_default_and_changes_the_images_when_on():
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    plain = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3)
    assert plain.image_aug is False                                  # eval/default path: NO crop
    augd = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                                image_aug=True, rng_seed=1)
    # A nonzero crop must actually change the produced frames.
    assert not torch.allclose(plain[0]["obs_img_96"], augd[0]["obs_img_96"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_faithful_crop.py -v`
Expected: FAIL — `ImportError` (`V_RANDOM`, `_sample_crop_offsets`, `_apply_crop` undefined) / unknown `image_aug` kwarg.

- [ ] **Step 3: Implement the crop**

In `prismatic/vla/datasets/libero_dataset.py`, add module-level (near the other constants):
```python
# Random-crop augmentation, ported from the ORIGINAL AsyncVLA
# (`prismatic/vla/datasets/lelan_dataset.py:113-114, 363-371`):
#     voffset = int(224.0 * v_random * random.random())
#     hoffset = int(224.0 * h_random * random.random())
#     PILbox  = (hoffset, voffset, 224 - hoffset, 224 - voffset)
# Offsets are FRACTIONS of the frame, so they carry over to our 256px LIBERO frames.
# The original also horizontally flips (and mirrors the actions); we do NOT -- mirroring
# 6-DoF EEF + gripper is unsafe (see the spec's deviation ledger).
V_RANDOM = 0.2
H_RANDOM = 0.1


def _sample_crop_offsets(h: int, w: int, rng: "np.random.RandomState") -> Tuple[int, int]:
    """One (voffset, hoffset) per SAMPLE -- shared by every image of that sample."""
    return int(h * V_RANDOM * rng.random()), int(w * H_RANDOM * rng.random())


def _apply_crop(raw: np.ndarray, voffset: int, hoffset: int) -> np.ndarray:
    """Symmetric crop, matching the original's PILbox = (h, v, W-h, H-v)."""
    if voffset == 0 and hoffset == 0:
        return raw
    h, w = raw.shape[:2]
    return raw[voffset : h - voffset, hoffset : w - hoffset]
```
In `__init__`, add `image_aug: bool = False` (store `self.image_aug = image_aug`), initialise `self._aug_rng = np.random.RandomState(rng_seed)` and `self._last_crop_offsets = (0, 0)`.

In `__getitem__`, immediately **before** the image lines (currently **377–387**), compute the single shared box and crop the four raw frames:
```python
        raw_now = ep["image"][t]
        raw_base = ep["image"][t_base]
        raw_wrist = ep["wrist_image"][t_base]
        raw_prev = raw_base if self.delay_aware else (ep["image"][t - 1] if t > 0 else raw_now)

        if self.image_aug:
            v, hh = _sample_crop_offsets(raw_now.shape[0], raw_now.shape[1], self._aug_rng)
            self._last_crop_offsets = (v, hh)
            raw_now = _apply_crop(raw_now, v, hh)
            raw_base = _apply_crop(raw_base, v, hh)
            raw_wrist = _apply_crop(raw_wrist, v, hh)
            raw_prev = _apply_crop(raw_prev, v, hh)
```
then build the images from these cropped arrays instead of re-indexing the episode:
```python
        primary_now = Image.fromarray(raw_now)
        primary_prev = Image.fromarray(raw_prev)
        base_primary_image = _to_base_image(raw_base)
        base_wrist_image = _to_base_image(raw_wrist)
```
(The existing `_to_base_image` resize-to-224 + center-crop and `_to_edge_frame` 96px resize run unchanged **after** the crop, so a cropped frame is simply resized as before.) Everything else in `__getitem__` — proprio, action chunks, `t_base` logic — is untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_faithful_crop.py tests/test_libero_dataset.py tests/test_libero_dataset_delay.py -v`
Expected: PASS (new crop tests + the existing dataset and delay tests still green — `image_aug` defaults to False, so the old paths are byte-identical).

- [ ] **Step 5: Commit**

```bash
git add prismatic/vla/datasets/libero_dataset.py tests/test_faithful_crop.py
git commit -m "faithful: random-crop augmentation (v=0.2, h=0.1), one shared box per sample"
```

---

## Task 4: Wire the faithful defaults into the trainer (`k_max=3`, capacity, aug)

**Files:**
- Modify: `vla-scripts/train_asyncvla_libero.py` (`AsyncVLALiberoConfig` defaults; dataset construction; `build_edge_and_proj` call; `train_one_batch_smoke`)
- Test: `tests/test_faithful_train_overfit.py`

**Interfaces:**
- Consumes: `EdgeArch.from_config_nav()` (Task 1); `faithful_chunk_loss` (Task 2); `LiberoSpatialDataset(..., image_aug=...)` (Task 3).
- Produces: config fields `k_max: int = 3`, `image_aug: bool = True`, and edge-arch fields defaulting to the config_nav values. `train_one_batch_smoke(num_iters=200, delay_aware=False, k_max=3, image_aug=False)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faithful_train_overfit.py
import pytest
pytestmark = pytest.mark.slow


def test_faithful_edge_overfits_tiny_batch():
    """Faithful config end-to-end: original Edge_adapter @1024/4/4, k_max=3, 3-term MSE."""
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    losses = train_one_batch_smoke(num_iters=200, delay_aware=True, k_max=3)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_faithful_train_overfit.py -v -m slow`
Expected: FAIL (before Step 3, `train_one_batch_smoke` still builds the 512/2/2 edge and the L1 loss; after Tasks 1–2 it may error on the removed `Edge_adapter_manip` import). Ensure the GPU is clear first.

- [ ] **Step 3: Wire the faithful defaults**

In `AsyncVLALiberoConfig`:
```python
    k_max: int = 3                          # ORIGINAL: lelan_dataset.py:304 -> randint(0, min(t, 3))
    image_aug: bool = True                  # ORIGINAL: random crop (v=0.2, h=0.1); train only
    # Edge capacity: the ORIGINAL config_nav values (1024/4/4/4), not the class defaults.
    edge_obs_encoding_size: int = 1024
    edge_mha_heads: int = 4
    edge_mha_layers: int = 4
    edge_mha_ff_dim_factor: int = 4
```
Build the arch from the config and pass it through:
```python
    arch = EdgeArch(
        obs_encoding_size=cfg.edge_obs_encoding_size,
        mha_num_attention_heads=cfg.edge_mha_heads,
        mha_num_attention_layers=cfg.edge_mha_layers,
        mha_ff_dim_factor=cfg.edge_mha_ff_dim_factor,
    )
    edge, proj = build_edge_and_proj(vla.llm_dim, device, arch)
```
(`save_training_checkpoint` already writes `edge_arch.json` from `arch`, so eval auto-detects 1024/4/4.)
Forward `image_aug=cfg.image_aug` into the training `LiberoSpatialDataset(...)`.
In `train_one_batch_smoke`, change the signature to `(num_iters: int = 200, delay_aware: bool = False, k_max: int = 3, image_aug: bool = False)`, build the dataset with `image_aug=image_aug`, and build edge+proj with `EdgeArch.from_config_nav()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_faithful_train_overfit.py -v -m slow`
Expected: PASS (final loss < half the initial). Also run the fast suite: `.venv/bin/python -m pytest tests/ -v -m "not slow"` — all green.
If the loss **diverges** (NaN/exploding), do NOT retune the weights: record the loss trajectory and report it as a finding (the spec's Risk 1 — the `7.5×` delta weight under a different embodiment).

- [ ] **Step 5: Commit**

```bash
git add vla-scripts/train_asyncvla_libero.py tests/test_faithful_train_overfit.py
git commit -m "faithful: k_max=3, edge 1024/4/4 from config_nav, random-crop aug on by default"
```

---

## Task 5: Cluster — faithful training, latency, cadence sweep, results

> Push first: `git push exp asyncvla-libero-faithful`. On the cluster, use a SEPARATE checkout so the running `asyncvla-libero` jobs are untouched:
> `git -C /work/roboleon1295/asyncvla_libero_ws clone -b asyncvla-libero-faithful <repo-url> AsyncVLA-LIBERO-faithful` (or `git fetch && git worktree add`), reusing the same `.venv`, HF cache and RLDS data.

**Files:**
- Create (cluster): `train_faithful.sbatch`, `eval_cadence_faithful.sbatch`
- Create: `docs/superpowers/notes/faithful-results.md`

**Interfaces:**
- Consumes: the pushed faithful code; `docs/superpowers/notes/nano4-handoff.md` (env vars, partitions, account `mst114563`).
- Produces: a faithful edge checkpoint (`runs_libero/...--faithful/{shead,proj}--50000_checkpoint.pt` + `edge_arch.json`); `SR(N)` for N∈{1,2,3,4,6,8}; `t_base`/`t_edge` at 1024/4/4; the faithful curve in `faithful-results.md`.

- [ ] **Step 1: Pilot the faithful training (dev partition, short)**

```bash
srun --account=mst114563 --partition=dev --gres=gpu:1 --cpus-per-task=12 --mem=100G --time=00:15:00 \
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_asyncvla_libero.py \
  --delay_aware True --k_max 3 --max_steps 20 --save_freq 10 --episode_limit 10 --run_id_note faithful
```
(with `HF_HOME`, `MUJOCO_GL=egl`, `WANDB_MODE=offline`.) Expected: 20 steps run; `shead--20_checkpoint.pt` **and `edge_arch.json`** written; the log shows the resolved arch is **1024/4/4/4**. Fix any runtime bug before the full run.

- [ ] **Step 2: Submit the full faithful training (8gpus, ~5h)**

Write `train_faithful.sbatch` (copy the Phase-1 `train.sbatch` pattern from `nano4-handoff.md`; add `--delay_aware True --k_max 3 --max_steps 50000 --save_freq 5000 --run_id_note faithful --wandb_project asyncvla-libero`). `sbatch train_faithful.sbatch`. Poll `squeue` / wandb until `shead--50000_checkpoint.pt` exists. Watch `train/mse_delta`, `train/mse_traj`, `train/mse_smooth` — if the loss diverges, report it (Risk 1), do not retune.

- [ ] **Step 3: Latency at the faithful capacity (dev partition)**

```bash
srun --account=mst114563 --partition=dev --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=00:20:00 \
  python -m experiments.robot.libero.latency_bench --obs_encoding_size 1024 --heads 4 --layers 4
```
Record `t_base_ms`, `t_edge_ms`, and the per-N speedup table into `faithful-results.md`.

- [ ] **Step 4: Faithful cadence sweep — N ∈ {1,2,3,4,6,8} (6 eval jobs)**

Write `eval_cadence_faithful.sbatch` parameterized by `--export=ALL,CADENCE=N`, running:
```
python -m experiments.robot.libero.run_libero_eval --mode async --base_cadence $CADENCE \
  --task_suite_name libero_spatial --num_trials_per_task 50 \
  --edge_ckpt $CKPT_DIR/shead--50000_checkpoint.pt --proj_ckpt $CKPT_DIR/proj--50000_checkpoint.pt \
  --wandb_project asyncvla-libero
```
(with the sim env vars incl. per-job `TMPDIR`; `CKPT_DIR` = the `--faithful` run dir). Submit N ∈ {1,2,3,4,6,8}. Confirm each log prints the resolved edge arch **1024/4/4** (auto-detected from `edge_arch.json`). Read each `SR(N)` from `slurm_logs/eval_cadence_faithful-*.log` (`Total episodes` / `Total successes`).

- [ ] **Step 5: Assemble the deliverable + commit**

Fill `docs/superpowers/notes/faithful-results.md` with: the `SR(N)` table for N∈{1,2,3,4,6,8} against the **0.982** stock topline; `t_base`/`t_edge` and per-N speedup; an SR-vs-N plot; and the **fidelity ledger** (spec §7) marking each original behavior *reproduced* / *embodiment-forced* / *compute-forced*. State plainly whether SR holds across the in-distribution regime (N ≤ 4) and where the out-of-distribution cliff (N = 6, 8) appears.

```bash
git add docs/superpowers/notes/faithful-results.md
git commit -m "docs: faithful AsyncVLA Phase-2 results (SR vs cadence N in {1,2,3,4,6,8}) + latency"
git push exp asyncvla-libero-faithful
```

---

## Self-Review

**Spec coverage:**
- §2 minimal-delta (call the original `Edge_adapter`; one-line change; retire the duplicate; read `config_nav`) → Task 1. ✔
- §3 fix 1 (capacity 1024/4/4) → Task 1 + Task 4. ✔ fix 2 (`k_max=3`) → Task 4. ✔ fix 3 (3-term MSE) → Task 2. ✔ fix 4 (random crop) → Task 3. ✔
- §4 loss (exact weights; gripper excluded from traj/smooth; `sm_ref` not detached; `obj_pose` dropped) → Task 2. ✔
- §5 training config (frozen base, AdamW 1e-4, 50k/batch 8, crop params, one shared box) → Tasks 3, 4, 5. ✔
- §6 async eval reused unchanged; sweep N∈{1,2,3,4,6,8} → Task 5 (no eval-harness edits anywhere in the plan — as intended). ✔
- §7 deviation ledger → reproduced verbatim in Task 5 Step 5's deliverable; the Global Constraints forbid inventing new deviations. ✔
- §9 testing (edge shape @config_nav; k_max=3 clamp — covered by the existing `test_libero_dataset_delay.py` sampler test, re-run in Task 3 Step 4; crop shared-box + off-at-eval; 3-term loss; delay-aware overfit; async smoke at N=3 — folded into Task 5 Step 1's pilot + Step 4's N=3 job). ✔
- §10 Risk 1 (do not retune the 7.5× weight; report divergence) → explicit in Task 4 Step 4 and Task 5 Step 2. ✔

**Placeholder scan:** No TBD/TODO. Every code step carries complete code. Cluster steps carry exact commands and expected outputs. The one judgement call left to the implementer — the cluster checkout method in Task 5's preamble (clone vs `git worktree add`) — is bounded and either is acceptable, with the binding requirement stated (must not disturb the running `asyncvla-libero` jobs).

**Type consistency:** `EdgeArch` field names (`obs_encoding_size`, `mha_num_attention_heads`, `mha_num_attention_layers`, `mha_ff_dim_factor`) are identical in Task 1 (`from_config_nav`, `build_edge_and_proj`) and Task 4 (config → arch). `build_edge_and_proj(llm_dim, device, arch) -> Tuple[Edge_adapter, Proj_Actiontokens]` matches its Task-4 call. `faithful_chunk_loss(pred, gt) -> (Tensor, {"loss","mse_delta","mse_traj","mse_smooth"})` (Task 2) matches the Task-2 wandb keys and the Task-4/5 monitoring. `_sample_crop_offsets(h, w, rng) -> (voffset, hoffset)` and `_apply_crop(raw, voffset, hoffset)` (Task 3) match their `__getitem__` call sites. `train_one_batch_smoke(num_iters, delay_aware, k_max, image_aug)` (Task 4) matches its Task-4 test. **Breaking change noted:** the metrics key `"l1_loss"` is removed in Task 2 — its only two consumers (`run_forward_pass`, the wandb log at line 275) are both updated in that same task.
