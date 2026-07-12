# AsyncVLA-LIBERO Design Spec

**Date:** 2026-07-12
**Status:** Draft for review
**Author:** Chung-Yi Li (with Claude Code)

## 1. Summary

Retarget AsyncVLA's asynchronous **base VLA + Edge Adapter** control structure from
vision navigation to **LIBERO-Spatial manipulation**, using a **frozen released
OpenVLA-OFT LIBERO** checkpoint as the base. This is a **parallel experiment** built
directly from the original AsyncVLA code (PyTorch / Prismatic / OpenVLA-OFT lineage),
deliberately *not* referencing the sibling `../async_pi0` attempt (openpi/JAX pi0.5),
which has been struggling.

**Success = the async thesis holds on LIBERO:** the fast Edge Adapter sustains a high
task **success rate (SR)** while cutting **per-step inference latency** versus running
the full base VLA every control step.

Built in a **new git worktree**; existing navigation code is left untouched and LIBERO
modules are added alongside it.

## 2. Background: what we are mirroring

AsyncVLA (paper arXiv:2602.13476, `2602.13476.pdf` in repo) is a **nested control
system**:

- **Base VLA** (large, ~8B) — *slow outer loop* on a workstation. Provides rich
  vision-language guidance as **action-token embeddings**.
- **Edge Adapter** (~100× smaller, EfficientNet-B0 based) — *fast inner loop* on the
  robot. Continuously refines actions at high frequency, incorporating the **most
  recent** observation to correct for the base's latency.

### Algorithm 1 (AsyncVLA inference), reproduced

Two loops run asynchronously:

- **Workstation loop (slow, `f_b` Hz):** receive a delayed observation `I_{t-k}` and
  timestamp `t-k`; run base `π_b` on `I_{t-k}` + goal/instruction inputs → action-token
  embeddings; send embeddings + timestamp `t-k` back.
- **Onboard loop (fast, `f_s > f_b` Hz):** capture current obs `I_t` with timestamp,
  store `(I_t, t)` in a buffer `B`, send `I_t` to the workstation. When new base
  embeddings arrive, timestamp-match to retrieve the exact **delayed frame `I_{t-k}`**
  the base processed; feed the Edge Adapter `(I_t, I_{t-k}, embeddings)`; compute the
  action chunk `{Δa_{t+i}}` and execute it (nav: via a PD controller → velocity
  commands).

**Key consequence for our design:** the Edge Adapter's two image inputs are **the
current frame `I_t`** and **the stale frame `I_{t-k}` the base actually saw** — not
merely "current + previous frame." The edge's job is to bridge the temporal gap between
a stale base feature and the current observation. This dictates both training (sample a
delay `k`) and eval (two-rate loop).

### Base → edge interface in the original code

- Base forward with `output_hidden_states=True` → last-layer hidden states → slice the
  action-token positions → `actions_hidden_states` of shape `(B, NUM_ACTIONS_CHUNK ·
  ACTION_DIM, D)`.
- `Proj_Actiontokens.predict_action(actions_hidden_states, ...)` projects these into the
  compact **`vla_feature`** consumed by the edge (`(B, NUM_ACTIONS_CHUNK, feature_dim)`).
- `Edge_adapter(obs_img, past_img, vla_feature)` fuses `[vla_feature (8 tokens) ⊕
  obs_enc(1) ⊕ cat_enc(1)]` through a small transformer decoder → `action_predictor`
  MLP → action chunk. EfficientNet-B0 encoders are initialized **from scratch**
  (`from_name`, not ImageNet-pretrained).

## 3. Fixed decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Base VLA | Released OpenVLA-OFT LIBERO checkpoint (`moojink/openvla-oft`), **frozen** |
| Trained parts | Edge Adapter + feature projector only |
| Benchmark | **LIBERO-Spatial** (10 tasks) first |
| Objective | Demonstrate async speedup thesis: SR retained + per-step latency reduced |
| Edge design | **Approach A** — edge predicts the **full** 8×7 action chunk; residual-edge is the documented fallback |
| Edge encoders | EfficientNet-B0, **init from scratch** (ImageNet init known to hurt LIBERO 96px) |
| Async algorithm | **Mirror Algorithm 1** (two-rate loop, delayed-frame timestamp matching) |
| Action space | 7-DoF: 6 EEF deltas + 1 gripper (`ACTION_DIM=7`, `NUM_ACTIONS_CHUNK=8`) |
| Loss | L1 on the action chunk (OpenVLA-OFT style) |
| Tracking | Weights & Biases (per project rule) |
| Compute | NCHC Nano4 / Nano5 GPU |

## 4. Architecture

Four components (only the last two are trained):

1. **Frozen base** — OpenVLA-OFT LIBERO. Input: agentview 224px + language instruction.
   Loaded via OpenVLA-OFT's standard loader (HF `AutoModelForVision2Seq` /
   `experiments/robot/openvla_utils.py`), **not** forced into the AsyncVLA
   `OpenVLAForActionPrediction_MMNv1` class (see §8 risk). Runs with
   `output_hidden_states=True`; we extract action-token hidden states.

2. **Feature projector** (trainable) — `Proj_Actiontokens`-derived. Maps
   `actions_hidden_states (B, 8·7, D)` → `vla_feature (B, 8, 512)`.

3. **Edge Adapter** (trainable) — `Edge_adapter` retargeted for manipulation:
   - `obs_encoder`: EfficientNet-B0 (3ch) on current frame `I_t` @ 96px.
   - `cat_encoder`: EfficientNet-B0 (6ch) on `concat(I_t, I_{t-k})` @ 96px.
   - decoder fuses `[vla_feature(8) ⊕ obs_enc(1) ⊕ cat_enc(1)]` →
     `action_predictor` → **8×7 action chunk** (parametrized output dim; nav's 2D
     `delta_to_pose` removed).

4. **LIBERO constants** — `ACTION_DIM=7`, `NUM_ACTIONS_CHUNK=8`. Nav-only fields
   (`POSE_DIM`, goal-pose/image, 9 modality IDs, distance head, PD/UTM) excluded from the
   LIBERO code path.

## 5. Data flow

### Training (base frozen, edge + projector trained)

Per sample from a LIBERO-Spatial demo at timestep `t`:

1. Sample a delay `k` (fixed or randomized within a small range, emulating base latency).
2. Base forward (`no_grad`) on delayed frame `I_{t-k}` + instruction →
   `actions_hidden_states` → projector → `vla_feature`.
3. Edge forward on `(I_t, I_{t-k}, vla_feature)` → predicted 8×7 chunk.
4. **L1 loss** vs the ground-truth action chunk at `t` (`{a_{t+i}}_{i=0..7}`), using
   OpenVLA-OFT normalization stats from the base checkpoint.
5. Backprop into edge + projector only.

DataLoader (`libero_dataset.py`) yields: base inputs (processed agentview + tokenized
instruction), current frame `I_t` @96px, delayed frame `I_{t-k}` @96px, GT 8×7 chunk,
and normalization stats. Built on the repo's existing RLDS/OXE plumbing (which already
contains LIBERO configs); source demos = LIBERO-Spatial RLDS (OpenVLA-OFT
`modified_libero_rlds`).

### Async eval (mirror Algorithm 1)

In the LIBERO sim (robosuite), emulate the two-rate loop:

- Maintain cached `(vla_feature, I_{t-k})` = the base's latest output and the frame it
  processed.
- **Slow outer loop:** every `R = round(f_s / f_b)` env steps, trigger a base recompute;
  the base consumes the frame at trigger time, and its result becomes available `k` steps
  later (models base latency). On completion, update the cache.
- **Fast inner loop:** every env step, run the edge on `(I_t, cached I_{t-k}, cached
  vla_feature)` → chunk → execute (manipulation: apply the action directly; no PD).
- Sweep the async knob = base frequency / delay (`R`, `k`), mirroring the paper's
  frequency setup.

**Baseline (synchronous):** run the full base VLA every control step (standard
OpenVLA-OFT LIBERO rollout). Measure SR + mean per-step latency.

**Reported metrics per configuration:** LIBERO-Spatial **success rate** and **mean
per-step wall-clock latency**; async vs. synchronous baseline. All logged to wandb.

## 6. New files (navigation code untouched)

- `prismatic/vla/constants_libero.py` — LIBERO action/chunk constants (`ACTION_DIM=7`).
- `prismatic/models/small_head.py` — add an `Edge_adapter` variant with parametrized
  (8×7) output for manipulation.
- `prismatic/vla/datasets/libero_dataset.py` — LIBERO-Spatial loader with delayed-frame
  `(I_t, I_{t-k})` sampling and GT chunk targets.
- `vla-scripts/train_asyncvla_libero.py` — frozen base; train edge + projector; L1 loss;
  DDP; wandb.
- `experiments/robot/libero/run_async_libero_eval.py` — Algorithm-1 two-rate async
  rollout + synchronous baseline; SR + latency; wandb.
- `inference/run_asyncvla_libero.py` *(optional)* — single-episode smoke test.

## 7. Testing

- **Unit:** edge forward → `(B, 8, 7)`; projector output shape `(B, 8, 512)`; dataloader
  sample shapes and delayed-frame alignment.
- **Integration:** overfit a tiny LIBERO batch (edge + projector) → loss decreases;
  latency microbench (base forward vs. edge forward).
- **Base sanity:** confirm the frozen base reproduces stock LIBERO-Spatial SR *before*
  attaching the edge (validates checkpoint loading + hidden-state extraction).
- **Smoke eval:** 1–2 LIBERO-Spatial episodes end-to-end through the two-rate loop.

## 8. Risks & mitigations

1. **Checkpoint format mismatch (primary risk).** The released OpenVLA-OFT LIBERO
   checkpoint is HF-format, whereas AsyncVLA loads a custom Prismatic
   `OpenVLAForActionPrediction_MMNv1`. *Mitigation:* load the base via the standard
   OpenVLA-OFT loader and adapt action-token hidden-state extraction to that model; gate
   progress on the §7 base-sanity SR check.
2. **Action-token hidden-state slicing.** The exact positions of action tokens in the
   OpenVLA-OFT LIBERO sequence must match how `actions_hidden_states` is sliced.
   *Mitigation:* reuse OpenVLA-OFT's own action-masking utilities; verify projected
   actions from `vla_feature` match the base's own action outputs on a batch.
3. **SR retention of full-chunk edge.** If Approach A's edge underperforms, fall back to
   the **residual edge** (edge predicts a correction to the base's projected chunk).
4. **Sim async fidelity.** The two-rate emulation approximates real asynchrony; a
   threaded "true-async" latency mode is a later enhancement, not part of this milestone.

## 9. Out of scope (this milestone)

- LIBERO-Object / -Goal / -Long / -90 / -10 (Spatial first).
- Finetuning or training the base VLA.
- Dual-camera (wrist) edge, residual edge (fallback only), threaded true-async runtime.
- Real-robot deployment.
