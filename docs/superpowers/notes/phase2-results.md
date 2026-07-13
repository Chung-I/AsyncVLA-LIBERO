# Phase 2 results — AsyncVLA-LIBERO async loop + speedup study

**Setup.** Frozen OpenVLA-OFT LIBERO-Spatial base + trainable Edge Adapter. Async eval is
CLOSED-LOOP (matching the original AsyncVLA): the edge runs EVERY env step and executes only
`actions[0]` (predict-8 / execute-1, receding horizon); the base's `vla_feature` is recomputed
every **N** env steps and held stale in between (staleness `k = 0..N-1` frames).
Protocol: LIBERO-Spatial, 10 tasks x 50 trials = **500 trials** per cell, NCHC Nano4 (H200).

Two edges, both trained 50k steps on the frozen base:
- **Naive** = Phase-1 edge (base always sees the CURRENT frame; `past = I_{t-1}`). No delay training.
- **Delay-aware** = Phase-2 edge, delayed-frame sampling `k ~ U{0..15}`: base processes `I_{t-k}`,
  edge sees `(current I_t, base's frame I_{t-k}, stale vla_feature)`, target = chunk at `t`.

## Success rate vs base cadence

| N | staleness (frames) | Naive edge | Delay-aware | Δ | per-step compute | speedup vs base-every-step |
|---|---|---|---|---|---|---|
| 1 | 0 | **0.912** (456/500) | 0.626 (313/500) | −0.286 | 54.6 ms | 0.80x |
| 2 | 0–1 | **0.854** (427/500) | 0.730 (365/500) | −0.124 | 32.6 ms | 1.35x |
| 4 | 0–3 | 0.644 (322/500) | **0.780** (390/500) | +0.136 | 21.6 ms | 2.03x |
| 8 | 0–7 | 0.198 (99/500) | **0.826** (413/500) | **+0.628** | 16.1 ms | **2.72x** |
| 16 | 0–15 | 0.004 (2/500) | **0.602** (301/500) | **+0.598** | 13.4 ms | 3.28x |

**Reference lines (synchronous, Phase 1, open-loop-8):** stock base topline **0.982**;
synchronous base+edge **0.972**.

**Latency (H200, `latency_bench`):** `t_base = 43.9 ms`, `t_edge = 10.6 ms`.
Per-step compute `= t_edge + t_base/N`; speedup `= t_base / (t_edge + t_base/N)`; asymptote
`t_base/t_edge ≈ 4.1x`. (On an RTX 5090 the base is slower — 94.4 ms vs 5.7 ms edge — so the
asymptote there is ~16x. The speedup is hardware-dependent; the *compute* saving on the base
is exactly N x regardless.)

## Findings

**1. Delay-aware training is what makes async viable.** The naive edge collapses monotonically
under staleness (0.912 -> 0.004): an edge trained on a FRESH base feature is unusable once the
base lags 8–16 frames. The delay-aware edge holds up, reaching **0.826 at N=8 with ~8x less base
compute** (2.72x wall-clock). At N=8 delayed-frame training is worth **+0.63 SR**.

**2. The curves cross at ~N=4, and the delay-aware curve is NON-MONOTONIC.** It is *worst* at
N=1 (0.626) and *peaks* at N=8 (0.826). Interpretation: the delay-aware edge learns to
*compensate for lag*; when there is no lag (N=1 => `k=0` always => `past == current`) that
correction is spurious and costs it ~0.29 vs the naive edge. Mid-cadence is its sweet spot; by
N=16 the lag is genuinely too large and SR falls again.

**3. Async trades SR for compute — it is not free.** The best async config (0.826 @ N=8) is
still ~15 points below the synchronous edge (0.972) and ~16 below the stock topline (0.982).
Honest headline: *~8x less base compute for ~15 points of SR*.

## The `k_max` question (most actionable)

The ORIGINAL AsyncVLA **does** train delay-aware — verified in its `lelan_dataset.py`:
```python
lt = random.randint(0, min(iv, 3))                 # random delay k in {0..3}
pixel_values = image_transform(<frame at iv - lt>) # BASE sees the DELAYED frame
p_image = <frame at iv - lt>                       # edge "past"    = the base's frame
c_image = <frame at iv>                            # edge "current"
```
So our Phase-2 scheme is faithful to the original — except the original uses **k ∈ {0..3}**
while we used **k ∈ {0..15}** (chosen to cover cadence up to N=16). Our delay distribution is
**5x wider**, which is the most likely cause of the weak low-cadence end: the edge is a
generalist over delays it rarely encounters.

Consequently our **Phase-1 "naive" edge is an ABLATION, not the paper's method** — the paper's
method is the delay-aware edge.

**Next ablation:** train with `k_max=3` (the original's setting) and evaluate N<=4. Expect it to
dominate the low-cadence end and to recover much of the gap to the 0.972 synchronous edge.

## Job map (Nano4)
- Delay-aware training: 180632 (50k steps, `--delay_aware True --k_max 15`).
- Naive sweep: 180679 (N1), 180680 (N2), 180681 (N4), 180639 (N8), 180640 (N16).
- Delay-aware sweep: 181191 (N1), 181192 (N2), 181193 (N4), 181205 (N8), 181195 (N16).
- wandb: https://wandb.ai/leon129506/asyncvla-libero

## Operational notes (cluster)
- Sim evals intermittently `SIGABRT` at first render (EGL device-0 contention with other jobs).
  It fails fast (~1 min) — **just resubmit**. Do NOT set `MUJOCO_EGL_DEVICE_ID`:
  `CUDA_VISIBLE_DEVICES` is remapped to 0, so a non-zero id fails to init and aborts at import.
- Parse SR with `grep -oE 'Total successes: *[0-9]+'` (log lines carry a `run_libero_eval.py:NNN`
  suffix that naive last-number parsing picks up instead).
