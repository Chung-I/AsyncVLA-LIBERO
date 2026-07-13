# AsyncVLA-LIBERO Phase 2 Design Spec — the async loop + speedup study

**Date:** 2026-07-13
**Status:** Draft for review
**Author:** Chung-Yi Li (with Claude Code)
**Builds on:** Phase 1 (`2026-07-12-asyncvla-libero-design.md`), branch `asyncvla-libero`.

## 1. Summary

Phase 1 ported AsyncVLA's frozen-base + Edge-Adapter structure to LIBERO-Spatial and showed
the **synchronous** edge (base + edge every step) reaches **SR 0.972** vs the stock base
topline **0.982** (500-trial protocol, NCHC Nano4 / H200).

Phase 2 demonstrates AsyncVLA's actual claim: a **fast edge running every step on a stale
base feature** sustains success while the **slow base runs only every N steps**, cutting
per-step compute ~N×. The deliverable is an **SR-vs-cadence(/latency) curve** showing a
**delay-aware** edge stays near the 0.972 synchronous baseline as the base cadence N grows,
while the Phase-1 (synchronous-trained) edge degrades — quantified against the base-every-step
topline (0.982).

Two additions to Phase 1: (a) **delay-aware training** so the edge learns to bridge a stale
feature; (b) a **cadence-N async evaluation**. Everything else is reused: frozen OpenVLA-OFT
LIBERO base, `Edge_adapter_manip`, `Proj_Actiontokens`, the LIBERO-Spatial dataset, and the
eval harness.

## 2. Background: what changes vs Phase 1

- **Phase 1 (synchronous).** Every step the base runs on the **current** frame `I_t` → fresh
  `vla_feature`; the edge takes `(I_t, I_{t-1}, fresh vla_feature)` → action chunk. Trained on
  consecutive frames with a fresh feature.
- **Phase 2 (asynchronous, cadence-N emulation).** The base runs only every N steps. At step
  `t`, the latest base output was computed at `t' = ⌊t/N⌋·N` on frame `I_{t'}`, so it is
  **stale by `k = t − t'` steps** (`k ∈ {0..N−1}`). The edge takes `(current I_t, the base's
  frame I_{t'}, stale vla_feature)` → action chunk, executed every step.

The Phase-1 edge never saw a stale feature (nor a >1-step frame gap) in training, so its SR is
expected to degrade as N grows. Phase 2 retrains the edge with **delayed-frame sampling** to
bridge the staleness, mirroring AsyncVLA Algorithm 1's timestamp-matched training.

## 3. Fixed decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Primary deliverable | **Async speedup curve**: SR vs base cadence N (and vs per-step latency), vs the 0.982 topline and 0.972 sync baseline |
| Staleness handling | **Retrain a delay-aware edge** (delayed-frame sampling) AND evaluate the **Phase-1 edge** as the degrading baseline |
| Async model in sim | **Cadence-N deterministic emulation** (base every N steps, edge every step on the latest stale feature) + **measured** `t_base`, `t_edge` for the latency axis |
| Training delay | `k ~ Uniform{0..K_max}`, **K_max = 15** (covers cadence up to 16); one edge for all cadences |
| Cadence sweep | **N ∈ {1, 2, 4, 8, 16}**, 500 trials each, for both edges |
| Latency | **Analytic** per-step compute `≈ t_edge + t_base/N`; speedup `= t_base / (t_edge + t_base/N)`. Threaded true-async deferred |
| Benchmark / base / recipe | LIBERO-Spatial; same frozen base; 50k-step / L1 / frozen-base training recipe as Phase 1 |
| Compute / tracking | NCHC Nano4 (H200); Weights & Biases |

## 4. Delay-aware training

Per training step, for a sampled timestep `t` in an episode:

1. Sample delay `k ~ Uniform{0..K_max}` (K_max=15); clamp so `t−k ≥ 0` (episode start).
2. Frozen base forward (`no_grad`) on the **delayed** frame `I_{t−k}` + instruction →
   `actions_hidden_states` → projector → **stale** `vla_feature`.
3. Edge forward on `(current I_t, delayed I_{t−k}, stale vla_feature)` → predicted 8×7 chunk.
4. **L1 loss** vs the GT action chunk at `t` (same normalization / gripper standardization /
   center-cropped base images as Phase 1).
5. Backprop into edge + projector only (base frozen).

This makes a **single** edge robust across the whole cadence sweep. `k=0` reproduces the
Phase-1 synchronous case, so the delay-aware training strictly generalizes Phase 1.

The dataset already loads full episodes into memory, so `I_{t−k}` is a cheap index back; the
delayed base image goes through the same center-crop as `I_t` (Phase-1 fix) before the base.

## 5. Cadence-N async evaluation

Add `--mode async --base_cadence N` to the eval harness. Rollout per episode:

- Maintain a cache `(vla_feature, base_frame)`.
- **Every N env steps** (and at step 0): run the base on the **current** frame → refresh the
  cache with the new `vla_feature` and the frame it saw.
- **Every step**: run the edge on `(current frame, cached base_frame, cached vla_feature)` →
  unnormalize (identical post-processing to Phase 1) → execute. Same open-loop chunk cadence as
  the `stock`/`edge` modes.
- `N=1` is exactly the Phase-1 synchronous edge (built-in sanity check).

Sweep **N ∈ {1,2,4,8,16}**, 500 trials each (10 tasks × 50), for **both** the delay-aware edge
and the Phase-1 edge, under the identical protocol used for the topline.

## 6. Metrics & the deliverable curve

- **SR(N)** per cadence, both edges (500-trial protocol), overall + per-task.
- **Latency microbench** (`latency_bench.py`, H200): `t_base` = one base forward (incl.
  hidden-state extraction + projector); `t_edge` = one edge forward. Per-step compute
  `≈ t_edge + t_base/N`; **speedup vs base-every-step** `= t_base / (t_edge + t_base/N)`.
- **Deliverable:** an SR-vs-N (and SR-vs-per-step-latency) plot — delay-aware edge staying flat
  near 0.972 while the Phase-1 edge degrades with N — annotated against the 0.982 topline.
  Numbers + plot logged to wandb and written to `docs/superpowers/notes/phase2-results.md`.

## 7. New / changed files (all additive; Phase-1 paths unchanged by default)

- `prismatic/vla/datasets/libero_dataset.py` — add delay-`k` sampling producing the delayed
  base frame + stale-feature training path. Default (k fixed at prev-frame) unchanged; new
  behavior behind a `delay_aware` / `k_max` option.
- `vla-scripts/train_asyncvla_libero.py` — `--delay_aware` (+ `--k_max`); the per-step forward
  runs the base on `I_{t−k}`.
- `experiments/robot/libero/edge_policy.py` + `run_libero_eval.py` — `--mode async
  --base_cadence N` (cache + hold logic; `EdgePolicy` gains the stale-feature path).
- `experiments/robot/libero/latency_bench.py` — measures `t_base`, `t_edge` on H200.
- Results/plot script → `docs/superpowers/notes/phase2-results.md`.

## 8. Testing

- **Unit:** delayed-frame sampler returns `I_{t−k}` for random k and clamps at episode start;
  async cache/hold logic (feature refreshes exactly every N steps; N=1 ≡ every step).
- **Integration:** delay-aware overfit test (loss drops on a tiny batch with random k); a
  1-episode async smoke at N=4 (pipeline runs end-to-end, valid 8×7 chunks).
- **Sanity:** async N=1 SR ≈ the Phase-1 `edge`-mode SR (equivalent action sequence).

## 9. Risks & mitigations

1. **Delay-aware edge may trail the Phase-1 edge at N=1** (generalist vs specialist).
   *Mitigation:* report both curves; if the gap is material, train a second edge with a
   narrower k range or upweight small k.
2. **Cadence-N ≠ real wall-clock async.** We report *emulated* staleness and *analytic* latency
   from measured `t_base`/`t_edge`, clearly labeled. Threaded true-async is out of scope here.
3. **Same cluster-runtime footprint as Phase 1** (fork/transformers, robosuite `/tmp`, EGL,
   DataLoader) — already solved and captured in `nano4-handoff.md`.

## 10. Out of scope (Phase 2)

- Threaded / real-wall-clock true-async runtime (a possible Phase 3).
- Robustness-to-perturbation study (AsyncVLA's other claim); residual-edge / dual-camera
  variants; LIBERO suites beyond Spatial.
