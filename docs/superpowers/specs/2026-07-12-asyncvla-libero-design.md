# AsyncVLA-LIBERO Design Spec

**Date:** 2026-07-12
**Status:** Draft for review
**Author:** Chung-Yi Li (with Claude Code)

## 1. Summary

Retarget AsyncVLA's **base VLA + Edge Adapter** structure from vision navigation to
**LIBERO-Spatial manipulation**, using a **frozen released OpenVLA-OFT LIBERO** checkpoint
as the base. This is a **parallel experiment** built directly from the original AsyncVLA
code (PyTorch / Prismatic / OpenVLA-OFT lineage), deliberately *not* referencing the
sibling `../async_pi0` attempt (openpi/JAX pi0.5), which has been struggling.

The north-star goal remains the async speedup thesis (fast edge sustains SR while cutting
per-step latency), but the work is **staged into two plans**:

- **Phase 1 (THIS spec): faithfully mirror the released AsyncVLA code on LIBERO.** Port
  the exact modules, training, and co-located inference behavior; retarget nav → 7-DoF
  manipulation and the frozen OpenVLA-OFT LIBERO base. Get it training and running on
  LIBERO-Spatial. **No two-rate async loop, no latency study.**
  **Acceptance bar:** our base+edge success rate must be **close to the stock
  OpenVLA-OFT-LIBERO topline SR** — so we run the stock model in the same LIBERO-Spatial
  sim and compare directly (see §5, §7).
- **Phase 2 (next plan, deferred): our interpretation of the paper.** Build the
  Algorithm-1 two-rate asynchronous loop (workstation/edge frequencies, delayed-frame
  timestamp matching), the latency/speedup measurement, and the frequency/cadence sweep.

Built in a **new git worktree** (`/home/chungyili/Codes/AsyncVLA-libero`, branch
`asyncvla-libero`); existing navigation code is left untouched and LIBERO modules are added
alongside it.

## 2. What the released AsyncVLA code actually contains (our source of truth for Phase 1)

The released repo ships **the model + training + a co-located single-PC inference demo** —
**not** the deployed asynchronous runtime.

- **Modules:** base VLA (`OpenVLAForActionPrediction_MMNv1`), `Proj_Actiontokens`
  projector, `Edge_adapter` (EfficientNet-B0 encoders, transformer decoder, MLP head).
- **Base → edge interface** (`inference/run_asyncvla.py::run_forward_pass`,
  `vla-scripts/train_asyncvla.py::run_forward_pass`):
  1. Base forward with `output_hidden_states=True`; slice action-token positions
     (`get_current_action_mask | get_next_actions_mask`) → `actions_hidden_states` of
     shape `(B, NUM_ACTIONS_CHUNK · ACTION_DIM, D)`.
  2. `Proj_Actiontokens.predict_action(actions_hidden_states, taskid)` → the compact
     **`vla_feature`** consumed by the edge (`(B, NUM_ACTIONS_CHUNK, ·)`).
  3. `Edge_adapter(obs_img, past_img, vla_feature)` fuses `[vla_feature (8 tokens) ⊕
     obs_enc(1) ⊕ cat_enc(1)]` through a small transformer decoder → `action_predictor`
     MLP → action chunk. `obs_img` = current frame, `past_img` = previous frame, both
     96px. EfficientNet-B0 encoders are initialized **from scratch** (`from_name`).
- **Co-located "async" demo:** base runs **once** → `vla_feature`; the edge runs in a
  `for i in range(2)` loop over two frames (`past.png`, then `cur.png`) **reusing that
  same `vla_feature`**. This is the async idea in miniature — *the edge re-predicts on a
  newer frame while the base feature stays stale* — but there are **no sockets, threads,
  timestamp buffers, or ROS** in the repo. The real two-rate deployment is described only
  in the paper (Algorithm 1) and runs on the robot via ROS1.

**Phase 1 mirrors the items above.** The two-rate loop, delayed-frame (`I_{t-k}`) sampling,
and latency measurement are *paper interpretation* and are **deferred to Phase 2**.

## 3. Fixed decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Base VLA | Released OpenVLA-OFT LIBERO checkpoint (`moojink/openvla-oft`), **frozen** |
| Trained parts | Edge Adapter + `Proj_Actiontokens` projector only |
| Benchmark | **LIBERO-Spatial** (10 tasks) first |
| North-star objective | Async speedup thesis (SR retained + per-step latency reduced) — **demonstrated in Phase 2** |
| Phase 1 objective | **Faithful code mirror on LIBERO**: modules ported, trains, runs functionally |
| Phase 1 success criterion | Base+edge SR **close to stock OpenVLA-OFT-LIBERO topline SR** (target: within ~5% absolute SR — confirm at review), same eval protocol |
| Topline baseline | Stock OpenVLA-OFT-LIBERO (native action head, no edge), evaluated in the same LIBERO-Spatial sim |
| Eval protocol | OpenVLA-OFT standard: 10 LIBERO-Spatial tasks × 50 rollouts = 500 trials, identical seeds/config for stock and ours |
| Edge design | **Approach A** — edge predicts the **full** 8×7 action chunk; residual-edge is a Phase-2 fallback |
| Edge encoders | EfficientNet-B0, **init from scratch** (mirrors released code; ImageNet init also known to hurt LIBERO 96px) |
| Action space | 7-DoF: 6 EEF deltas + 1 gripper (`ACTION_DIM=7`, `NUM_ACTIONS_CHUNK=8`) |
| Loss | Action-chunk regression (L1/MSE), mirroring the released loss minus nav-only terms |
| Tracking | Weights & Biases (per project rule) |
| Compute | NCHC Nano4 / Nano5 GPU |

## 4. Architecture (Phase 1)

Four components; only the last two are trained:

1. **Frozen base** — OpenVLA-OFT LIBERO. Input: agentview 224px + language instruction.
   Loaded via OpenVLA-OFT's standard loader (HF `AutoModelForVision2Seq` /
   `experiments/robot/openvla_utils.py`), **not** forced into AsyncVLA's
   `OpenVLAForActionPrediction_MMNv1` (see §8 risk). `output_hidden_states=True`; extract
   action-token hidden states via OpenVLA-OFT's action masking.

2. **Feature projector** (`Proj_Actiontokens`, trainable) — maps `actions_hidden_states
   (B, 8·7, D)` → `vla_feature (B, 8, ·)`. `taskid`/modality is fixed to a single constant
   for LIBERO (single modality: language) — mirrors the code's interface without the 9-way
   nav modality logic.

3. **Edge Adapter** (`Edge_adapter` retargeted, trainable) — identical structure to the
   released module, output dim parametrized from 8×4 → **8×7**:
   - `obs_encoder`: EfficientNet-B0 (3ch) on current frame @ 96px.
   - `cat_encoder`: EfficientNet-B0 (6ch) on `concat(current, previous)` @ 96px.
   - decoder fuses `[vla_feature(8) ⊕ obs_enc(1) ⊕ cat_enc(1)]` → `action_predictor` →
     **8×7 action chunk**. Nav's 2D `delta_to_pose` and PD controller are removed
     (manipulation actions are applied directly).

4. **LIBERO constants** — `ACTION_DIM=7`, `NUM_ACTIONS_CHUNK=8`. Nav-only fields
   (`POSE_DIM`, goal-pose/image, 9 modality IDs, distance head) excluded from the LIBERO
   path.

## 5. Data flow (Phase 1)

### Training (base frozen; edge + projector trained) — mirrors `train_asyncvla.py`

Per sample from a LIBERO-Spatial demo at timestep `t`:

1. Base forward (`no_grad`) on the current agentview frame + instruction →
   `actions_hidden_states` → projector → `vla_feature`.
2. Edge forward on `(current frame I_t, previous frame I_{t-1}, vla_feature)` → predicted
   8×7 chunk. **(Consecutive frames, mirroring the code — no paper delay-`k` sampling in
   Phase 1.)**
3. **Loss** = action-chunk regression (L1/MSE) vs the GT chunk `{a_{t+i}}_{i=0..7}`, using
   OpenVLA-OFT normalization stats from the base checkpoint. Nav-only loss terms (distance,
   object pose, LeLaN language) are dropped.
4. Backprop into edge + projector only. DDP + wandb retained from the original script.

DataLoader (`libero_dataset.py`) yields: base inputs (processed agentview + tokenized
instruction), current frame @96px, previous frame @96px, GT 8×7 chunk, normalization
stats. Built on the repo's existing RLDS/OXE plumbing (which already contains LIBERO
configs); source demos = LIBERO-Spatial RLDS (OpenVLA-OFT `modified_libero_rlds`).

### Evaluation (Phase 1) — topline comparison in LIBERO-Spatial sim

The same eval harness runs **two configurations** in the LIBERO-Spatial sim under an
**identical protocol** (10 tasks × 50 rollouts = 500 trials, same seeds/config):

1. **Topline baseline — stock OpenVLA-OFT-LIBERO.** The base model produces actions via
   its **native** action head (OpenVLA-OFT parallel decoding + L1 continuous head), no
   edge. Yields `SR_topline`. This is both the reference number and the base-loading
   sanity check (we should reproduce the published stock SR).
2. **Ours — AsyncVLA-LIBERO (base + edge).** Same frozen base; actions come from the
   **edge adapter**, mirroring the released `run_forward_pass` (base → projector →
   `vla_feature`; edge on current + previous frame + `vla_feature` → chunk). Rollout is
   **synchronous** (base + edge every control step, edge's action executed). Yields
   `SR_ours`.

**Phase-1 success:** `SR_ours` is close to `SR_topline` (target within ~5% absolute SR —
confirm at review). Report both SRs (overall + per-task) side by side to wandb.

**Explicitly NOT in Phase 1:** decoupling base/edge rates, running base less often than
edge, timestamp buffers, latency measurement, cadence/frequency sweep. Those are Phase 2
(the synchronous base+edge rollout here is a functional/quality comparison, not a speed
claim).

## 6. New files (Phase 1; navigation code untouched)

- `prismatic/vla/constants_libero.py` — LIBERO action/chunk constants (`ACTION_DIM=7`).
- `prismatic/models/small_head.py` — add an `Edge_adapter` variant with parametrized (8×7)
  output for manipulation.
- `prismatic/vla/datasets/libero_dataset.py` — LIBERO-Spatial loader (current + previous
  frame, GT chunk targets).
- `vla-scripts/train_asyncvla_libero.py` — mirror of `train_asyncvla.py`: frozen base;
  train edge + projector; action-chunk loss; DDP; wandb.
- `experiments/robot/libero/run_libero_eval.py` — LIBERO-Spatial rollout supporting **two
  modes** under one protocol: `stock` (native OpenVLA-OFT head → `SR_topline`) and `edge`
  (our base+edge → `SR_ours`); reports overall + per-task SR side by side to wandb.
- `inference/run_asyncvla_libero.py` *(optional)* — single-episode smoke test mirroring the
  released demo forward pass.

## 7. Testing (Phase 1)

- **Unit:** edge forward → `(B, 8, 7)`; projector output shape; dataloader sample shapes.
- **Integration:** overfit a tiny LIBERO batch (edge + projector) → loss decreases.
- **Topline reproduction (gates everything):** run the stock OpenVLA-OFT-LIBERO model in
  the sim and confirm it reproduces the published stock LIBERO-Spatial SR. This validates
  checkpoint loading + the eval harness and establishes `SR_topline` before the edge is
  attached.
- **Smoke eval:** 1–2 LIBERO-Spatial episodes end-to-end through the base+edge rollout.
- **Final comparison:** full 500-trial protocol for both `stock` and `edge` modes; check
  `SR_ours` vs `SR_topline` against the acceptance bar.

## 8. Risks & mitigations (Phase 1)

1. **Checkpoint format mismatch (primary risk).** The released OpenVLA-OFT LIBERO
   checkpoint is HF-format, whereas AsyncVLA loads a custom Prismatic
   `OpenVLAForActionPrediction_MMNv1`. *Mitigation:* load the base via the standard
   OpenVLA-OFT loader and adapt action-token hidden-state extraction to that model; gate
   progress on the §7 base-sanity SR check.
2. **Action-token hidden-state slicing.** Action-token positions in the OpenVLA-OFT LIBERO
   sequence must match how `actions_hidden_states` is sliced. *Mitigation:* reuse
   OpenVLA-OFT's own action-masking utilities; verify projected actions from `vla_feature`
   track the base's own action outputs on a batch.
3. **SR retention of full-chunk edge.** If the full-chunk edge underperforms, the residual
   edge (edge corrects the base's projected chunk) is a Phase-2 fallback.

## 9. Phase 2 (deferred to the next plan — our paper interpretation)

Listed here only to scope Phase 1; **not designed in this spec.**

- Algorithm-1 two-rate asynchronous loop: base VLA at `f_b`, edge at `f_s > f_b`.
- Delayed-frame `I_{t-k}` sampling in training + timestamp-matched retrieval at inference.
- Latency/speedup measurement; base-frequency / delay / cadence sweep; SR-vs-latency
  Pareto vs. the synchronous baseline.
- Optional: threaded "true-async" runtime; residual-edge and dual-camera (wrist) variants;
  additional LIBERO suites (Object/Goal/Long).

## 10. Out of scope (both phases, this experiment)

- Finetuning or training the base VLA.
- Real-robot deployment.
- LIBERO suites beyond Spatial for the initial milestones (Spatial first).
