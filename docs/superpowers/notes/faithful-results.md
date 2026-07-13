# Faithful AsyncVLA reproduction — results

**Date:** 2026-07-14
**Code:** `asyncvla-libero-faithful` @ `3b7958d` (verified: the training and eval jobs log their
commit and assert `prismatic` resolves to the faithful worktree, not the sibling checkout).
**Setup:** LIBERO-Spatial, frozen OpenVLA-OFT 7B base, edge + projector trained 50k steps
(batch 8, AdamW lr 1e-4, `MultiStepLR` 10× decay at 25k). Eval: 500 trials per cadence
(10 tasks × 50), `--mode async`, closed-loop (edge every step, execute `action[0]`).
**Baseline** = the previous reproduction (`asyncvla-libero`), same 50k steps / same eval protocol.

## Success rate vs base cadence N

| N | baseline (previous repro) | **faithful** | Δ |
|---|---|---|---|
| 1 | 0.626 | **0.890** | **+26.4** |
| 2 | 0.730 | **0.934** | **+20.4** |
| 4 | 0.780 | **0.934** | **+15.4** |
| 8 | 0.826 | **0.886** | **+6.0** |
| 16 | **0.602** | 0.266 | **−33.6** |

Peak: **faithful 0.934 @ N=2–4** vs baseline 0.826 @ N=8 (**+10.8** best-vs-best).
At equal compute for every N ≤ 8, faithful strictly dominates.

## Latency at the faithful capacity (H200)

`t_base = 44.9 ms`, `t_edge = 11.2 ms` (1024/4/4/4, 76.5M-param edge).

| N | per-step compute (`t_base/N + t_edge`) | speedup vs base-every-step |
|---|---|---|
| 1 | 56.1 ms | 0.80× |
| 2 | 33.6 ms | 1.34× |
| 4 | 22.4 ms | 2.00× |
| 8 | 16.8 ms | 2.67× |
| 16 | 14.0 ms | 3.21× |

**Fidelity is nearly free.** The faithful edge has 3.6× the parameters of the baseline's
(76.5M vs 21.5M) but costs only ~5% more per edge tick (11.18 vs 10.65 ms). The 512/2/2 edge
was giving up most of the model's capacity to buy almost nothing.

## The finding: `k_max` selects the operating regime

The N=16 collapse is **not a bug — it is the mechanism**.

The faithful model trains with `k_max=3` (the original's value: `lelan_dataset.py:305`,
`randint(0, min(iv, 3))`), so it never sees a base feature staler than 3 steps. At N=16 the
edge must bridge up to 15 steps of staleness — far outside its training distribution — and it
collapses (0.266). The baseline trained with `k_max=15`, which is matched to N=16, and holds
up (0.602).

So the two models are not simply better/worse: **the training delay distribution sets the
cadence regime the edge works in.** `k_max` is not a minor hyperparameter — it is the knob
that picks the operating point. The previous reproduction's `k_max=15` was accidentally
buying high-cadence robustness at the cost of everything below it.

Practical consequence: the faithful curve is *flat* at 0.934 across N=2–4, so you can refresh
the base more often — better SR **and** less staleness — with no SR penalty. Deploy at N=4
(0.934 SR, 2.0× compute win) or N=8 (0.886, 2.67×).

## Caveat: this is a multi-factor comparison

Five things changed together (edge capacity 512/2/2→1024/4/4; `k_max` 15→3 **and** its
sampling distribution; L1 loss → the original's weighted delta+trajectory MSE; random-crop
augmentation added; `MultiStepLR` restored). The N=16 result isolates `k_max` fairly
convincingly, but the *magnitude* of the N≤8 gains cannot be attributed to any single factor
from this data. An ablation is the obvious follow-up.

## Deliberate deviations from the original (see the spec's §7 ledger)

- **Smoothness term dropped.** Proved their 4th term collapses to
  `¼·mean_t[dx² + dy² + 4sin²(Δθ/2)]` — a pure magnitude penalty on the deltas, no ground
  truth, no curvature (the SE(2) rotation is norm-preserving, so it cancels). It is "take
  small steps," not "don't jerk." Under LIBERO's offset (`bounds_q99`) normalization it is
  worse than inert: it biases toward a constant raw drift. A true smoothness (jerk) term is
  deferred as an ablation.
- `MultiStepLR` restored (neither LIBERO branch had any scheduler); fp32 rather than bf16;
  trajectory-term rotation integration is a small-angle approximation.
- The **async runtime is entirely ours** — the original never released a two-rate runtime
  (`inference/run_asyncvla.py` has no sockets/threads/queues and its loop breaks after one
  tick). Nothing to be faithful to there; say so in the paper.
