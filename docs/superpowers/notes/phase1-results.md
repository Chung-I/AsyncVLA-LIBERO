# Phase 1 results — AsyncVLA-LIBERO

## Stock LIBERO-Spatial topline (`SR_topline`) — Task 3.2

**Status: PLACEHOLDER.** The gating `SR_topline` number (LIBERO-Spatial, 10 tasks x 50
rollouts/task = 500 trials, `--mode stock`) is produced by the controller's reduced local
sweep / the full NCHC Nano4 run, **not** by this task. Task 3.2's job was wiring the
wandb logging + aggregation code and proving it with a short confirmation run (below).

Sanity gate (per the Task 3.2 brief): `SR_topline` should land in the published
OpenVLA-OFT LIBERO-Spatial range (~0.9+). If the controller's sweep comes in far below
that, STOP and debug base loading / prompt / normalization before validating the edge
against it.

Placeholder table to be filled in by the controller's sweep:

| task_id | task description | trials | SR |
|---|---|---|---|
| 0 | (libero_spatial task 0) | — | — |
| 1 | (libero_spatial task 1) | — | — |
| ... | ... | — | — |
| 9 | (libero_spatial task 9) | — | — |
| **overall** | **SR_topline** | **500** | **—** |

wandb project: `asyncvla-libero` (run name pattern `stock-libero_spatial-<n>trials`).

### Confirmation run (not the gate)

Ran the Task 3.2 confirm command to prove the wandb wiring + aggregation code works
end-to-end (2 tasks x 2 trials = 4 episodes, `libero_spatial`, `--mode stock`):

```
MUJOCO_GL=egl .venv/bin/python -m experiments.robot.libero.run_libero_eval \
  --mode stock --task_suite_name libero_spatial \
  --num_trials_per_task 2 --num_tasks 2 --wandb_project asyncvla-libero
```

| task_id | task description | trials | SR |
|---|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 2 | 0.50 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 2 | 0.00 |
| **overall** | (4-episode confirm run) | **4** | **0.25** |

This is a 4-episode smoke number only (not statistically meaningful, not the gate) —
it confirms per-task and overall SR are correctly computed and logged to wandb.

wandb run: https://wandb.ai/leon129506/asyncvla-libero/runs/qmreehhx

## Stock topline — reduced local sweep (RTX 5090, 100 trials)

**Date:** 2026-07-12. **Run completed cleanly** (exit 0, 100/100 episodes, ~13 min wall-clock).

This is a reduced local sweep (10 tasks x 10 trials = 100 trials, `--mode stock`,
`libero_spatial`) run on the local RTX 5090 — **not** the full 500-trial gate, but a
first real topline read.

```
MUJOCO_GL=egl .venv/bin/python -m experiments.robot.libero.run_libero_eval \
  --mode stock --task_suite_name libero_spatial \
  --num_trials_per_task 10 --num_tasks 10 \
  --wandb_project asyncvla-libero
```

**Overall success rate: 0.6100 (61/100 successes, 100 episodes).**

| task_id | task description | trials | SR |
|---|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 10 | 0.7000 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 10 | 0.1000 |
| 2 | pick up the black bowl from table center and place it on the plate | 10 | 1.0000 |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 10 | 1.0000 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 10 | 0.4000 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 10 | 0.4000 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 10 | 0.8000 |
| 7 | pick up the black bowl on the stove and place it on the plate | 10 | 0.5000 |
| 8 | pick up the black bowl next to the plate and place it on the plate | 10 | 0.9000 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 10 | 0.3000 |
| **overall** | **(100-trial reduced local sweep)** | **100** | **0.6100** |

wandb run: https://wandb.ai/leon129506/asyncvla-libero/runs/tvusfvhj
(run name `stock-libero_spatial-10trials`)

**Caveat vs. the gate:** 0.61 is well below the published OpenVLA-OFT LIBERO-Spatial
topline (~0.9+). Per-task SRs are noisy at only 10 trials/task, but the aggregate gap is
large enough to flag per the Task 3.2 sanity gate — if the full sweep confirms this, debug
base loading / prompt / normalization before validating the edge against it.
