# AsyncVLA-LIBERO Phase 2 Implementation Plan — async loop + speedup study

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show that a fast edge running every step on a *stale* base feature sustains LIBERO-Spatial success while the slow base runs only every N steps — an SR-vs-cadence(/latency) curve where a **delay-aware** edge stays near the 0.972 synchronous baseline as N grows, versus the Phase-1 edge that degrades.

**Architecture:** Reuse the entire Phase-1 stack (frozen OpenVLA-OFT LIBERO base, `Edge_adapter_manip`, `Proj_Actiontokens`, `LiberoSpatialDataset`, `train_asyncvla_libero.py`, `run_libero_eval.py`, `edge_policy.py`, `base_features.py`). Add two things: (1) **delay-aware training** — the frozen base processes a delayed frame `I_{t−k}` (with its own timestep's inputs), the edge takes `(current I_t, base's frame I_{t−k}, stale vla_feature)` → predicts the GT chunk at `t`; (2) a **cadence-N async eval** — the base feature is recomputed every N steps and held, the edge runs every step on the cached stale feature. Latency reported analytically from measured `t_base`/`t_edge`.

**Tech Stack:** PyTorch, HuggingFace Transformers (OpenVLA-OFT fork), EfficientNet-PyTorch, LIBERO + robosuite, RLDS/TFDS, Weights & Biases, DDP/torchrun, Slurm on NCHC Nano4 (H200).

## Global Constraints

- Work only in the worktree `/home/chungyili/Codes/AsyncVLA-libero` (branch `asyncvla-libero`). Do NOT touch `/home/chungyili/Codes/async_pi0`.
- **All Phase-1 behavior stays the default and unchanged.** Every Phase-2 code path is behind a new flag/argument; existing tests must keep passing.
- Local dev + smoke on the RTX 5090; full training + all cadence-sweep evals on **NCHC Nano4 (H200)** per `docs/superpowers/notes/nano4-handoff.md`. Push to GitHub `exp` remote (`git@github.com:Chung-I/AsyncVLA-LIBERO.git`), pull on the cluster.
- Interpreter: `.venv/bin/python` (no `python` on PATH). Cluster interpreter: `/work/roboleon1295/asyncvla_libero_ws/AsyncVLA-LIBERO/.venv/bin/python`.
- **Only one GPU model-loading job at a time** on a shared node; on the cluster always `export HF_HOME=/work/roboleon1295/.cache/huggingface MUJOCO_GL=egl TMPDIR=/work/roboleon1295/tmp/job_$SLURM_JOB_ID` (create TMPDIR) for sim jobs; the robosuite `/tmp` log fix + transformers fork are already in the repo/env.
- Constants: `ACTION_DIM=7`, `NUM_ACTIONS_CHUNK=8`, `num_patches=513`, `use_film=False`, base `llm_dim=4096`.
- Cadence sweep: **N ∈ {1, 2, 4, 8, 16}**. Delay-aware training: `k ~ Uniform{0..K_max}`, **K_max=15**. Eval protocol: 10 tasks × 50 trials = **500** per config.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01TNfYx4NdAAHHGzyKWqLtn6`
- Read eval SR from the run log (not the exit code): sim jobs segfault at teardown *after* results are computed/logged.

## Delay/async semantics (read once — every task depends on this)

- **Timestep `t`** in an episode with `k` sampled in `{0..min(K_max, t)}`:
  - The **base** processes the **delayed** timestep `t−k`: its image/wrist/proprio and its
    teacher-forced action-token labels are all built from `t−k` (exactly the Phase-1
    construction, but at `t−k` instead of `t`).
  - The **edge** input is `obs_img_96 = I_t` (current frame), `past_img_96 = I_{t−k}` (the
    frame the base processed — NOT `t−1`), and the **stale** `vla_feature` from the base.
  - The **target** `gt_action_chunk` is the chunk at `t` (what to do now), same
    normalization/gripper-standardization/center-crop as Phase 1.
  - `k=0` ⇒ base and edge both at `t`, `past_img_96 == obs_img_96`.
- **Async eval at cadence N:** cache `(vla_feature, base_frame)`. Every N steps (and step 0)
  recompute the base on the current frame and refresh the cache. Every step the edge runs on
  `(current frame, cached base_frame, cached vla_feature)`. `N=1` ⇒ base every step ⇒ identical
  to Phase-1 `--mode edge`.

---

## Milestone A — Delay-aware training path

### Task 1: Delay-`k` sampling in `LiberoSpatialDataset`

**Files:**
- Modify: `prismatic/vla/datasets/libero_dataset.py` (`__init__` signature, `__getitem__`)
- Test: `tests/test_libero_dataset_delay.py`

**Interfaces:**
- Consumes: existing `LiberoSpatialDataset(split, max_samples, data_dir, processor, episode_limit, predict_stop_token)`; helpers `_to_base_image`, `_to_edge_frame`, `_apply_libero_action_transform`, `build_base_item`, `_load_real_episodes`; `NUM_ACTIONS_CHUNK`, `ACTION_DIM`.
- Produces: `LiberoSpatialDataset(..., delay_aware: bool = False, k_max: int = 15, rng_seed: int = 0)`. When `delay_aware=False` the item dict is byte-for-byte the Phase-1 item. When `True`, `__getitem__` samples `k = min(randint(0, k_max), t)` and returns the same dict keys but with **base inputs built from `t−k`**, `past_img_96 = I_{t−k}`, `obs_img_96 = I_t`, `gt_action_chunk` = chunk at `t`. Also add a module-level helper `_sample_delay(t: int, k_max: int, rng: np.random.RandomState) -> int` returning `min(rng.randint(0, k_max + 1), t)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_libero_dataset_delay.py
import numpy as np
import torch
from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset, _sample_delay
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


def test_sample_delay_clamps_at_episode_start():
    rng = np.random.RandomState(0)
    # t < k_max must clamp to <= t; t=0 always yields 0.
    assert _sample_delay(0, 15, rng) == 0
    for _ in range(50):
        assert 0 <= _sample_delay(3, 15, rng) <= 3


def test_delay_aware_item_has_same_keys_and_shapes():
    # Uses the small local RLDS shard fixture path (skips if absent).
    from prismatic.vla.datasets.libero_dataset import DEFAULT_DATA_DIR, _has_tfrecords
    import pytest
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    ds = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=15)
    item = ds[0]
    assert item["obs_img_96"].shape == (3, 96, 96)
    assert item["past_img_96"].shape == (3, 96, 96)
    assert item["gt_action_chunk"].shape == (NUM_ACTIONS_CHUNK, ACTION_DIM)
    for key in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio"):
        assert key in item


def test_delay_aware_k0_past_equals_obs():
    # With k forced to 0, past_img_96 == obs_img_96 (base frame == current frame).
    from prismatic.vla.datasets.libero_dataset import DEFAULT_DATA_DIR, _has_tfrecords
    import pytest
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    ds = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=0)
    item = ds[0]
    assert torch.allclose(item["obs_img_96"], item["past_img_96"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_libero_dataset_delay.py -v`
Expected: FAIL — `_sample_delay` not defined / `delay_aware` kwarg unknown.

- [ ] **Step 3: Implement delay sampling**

Read the current `__getitem__` (around lines 347–390) and the `__init__` signature (around line 302) first. Then:

- Add `_sample_delay` at module level:
```python
def _sample_delay(t: int, k_max: int, rng: "np.random.RandomState") -> int:
    """Delay k for delay-aware training: Uniform{0..k_max}, clamped so t-k >= 0."""
    return int(min(rng.randint(0, k_max + 1), t))
```
- Add `delay_aware: bool = False, k_max: int = 15, rng_seed: int = 0` to `__init__`; store them; create `self._delay_rng = np.random.RandomState(rng_seed)` when `delay_aware`.
- In `__getitem__`, after `ep_idx, t = self._index[idx]`, compute the **base timestep**:
```python
if self.delay_aware:
    k = _sample_delay(t, self.k_max, self._delay_rng)
    t_base = t - k
else:
    t_base = t                       # Phase-1 behavior unchanged
```
  Build the base inputs (`base_primary_image`, `base_wrist_image`, proprio, and the
  teacher-forced action tokens via `build_base_item`) from **`t_base`** instead of `t` — i.e.
  replace the `t` used for base image/wrist/state/action-token construction with `t_base`.
  Keep the **edge target** and current frame at `t`:
```python
primary_now = ep["image"][t]                       # current frame I_t (edge obs)
primary_base = ep["image"][t_base]                 # base's frame I_{t-k} (edge past)
item["obs_img_96"] = _to_edge_frame(_pil(primary_now))
item["past_img_96"] = _to_edge_frame(_pil(primary_base))
# gt_action_chunk stays the chunk at t (target "now"), transformed as in Phase 1.
```
  (Match the exact PIL/np conversion the file already uses for `_to_edge_frame`; if the
  current code resizes `ep["image"][t]` a particular way, reuse that helper verbatim for both
  `primary_now` and `primary_base`.) When `delay_aware=False`, none of this runs — the original
  `t`-based path is taken and `past_img_96` stays `I_{t-1}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_libero_dataset_delay.py tests/test_libero_dataset.py -v`
Expected: PASS (new delay tests + all Phase-1 dataset tests still green).

- [ ] **Step 5: Commit**

```bash
git add prismatic/vla/datasets/libero_dataset.py tests/test_libero_dataset_delay.py
git commit -m "feat: delay-aware sampling in LiberoSpatialDataset (base at t-k, edge past=I_{t-k})"
```

### Task 2: `--delay_aware` / `--k_max` in the training script

**Files:**
- Modify: `vla-scripts/train_asyncvla_libero.py` (`AsyncVLALiberoConfig`, dataset construction, `train_one_batch_smoke`)
- Test: `tests/test_train_overfit_delay.py`

**Interfaces:**
- Consumes: `AsyncVLALiberoConfig`, `build_edge_and_proj`, `run_forward_pass`, `LiberoSpatialDataset`, `train_one_batch_smoke` from Phase 1.
- Produces: config fields `delay_aware: bool = False` and `k_max: int = 15`; the training `LiberoSpatialDataset(...)` call forwards them; `train_one_batch_smoke(num_iters=200, delay_aware=False, k_max=15)` gains the two kwargs and builds its dataset with them. The per-step `run_forward_pass` is UNCHANGED — it already consumes `obs_img_96`/`past_img_96`/`gt_action_chunk`/base inputs from the batch, which now carry the delayed-frame content.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_overfit_delay.py
import pytest
pytestmark = pytest.mark.slow


def test_delay_aware_edge_overfits_tiny_batch():
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    # Random k in {0..15} across 200 iters must still drive loss well below half.
    losses = train_one_batch_smoke(num_iters=200, delay_aware=True, k_max=15)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_overfit_delay.py -v -m slow`
Expected: FAIL — `train_one_batch_smoke` has no `delay_aware` kwarg.

- [ ] **Step 3: Implement the flags**

- In `AsyncVLALiberoConfig` add (near `episode_limit`):
```python
    delay_aware: bool = False               # Phase-2 delay-aware training (base sees I_{t-k}).
    k_max: int = 15                         # Max delay k ~ Uniform{0..k_max} when delay_aware.
```
- In the training `LiberoSpatialDataset(...)` construction, forward `delay_aware=cfg.delay_aware, k_max=cfg.k_max`.
- In `train_one_batch_smoke`, add params `delay_aware: bool = False, k_max: int = 15` and pass them to its `LiberoSpatialDataset(...)`. Note: with `delay_aware=True` the smoke batch mixes random-k samples; the overfit target is still the fixed batch, so 200 iters (Phase-1 robust count) clears the 0.5 bar.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_overfit_delay.py tests/test_train_overfit.py -v -m slow`
Expected: PASS (delay-aware overfit + Phase-1 overfit both green). Confirm no other model process is running first (`pgrep -f run_libero_eval`).

- [ ] **Step 5: Commit**

```bash
git add vla-scripts/train_asyncvla_libero.py tests/test_train_overfit_delay.py
git commit -m "feat: --delay_aware/--k_max training path (Phase-2 delayed-frame edge)"
```

---

## Milestone B — Cadence-N async evaluation

### Task 3: Async cache/hold in `EdgePolicy`

**Files:**
- Modify: `experiments/robot/libero/edge_policy.py` (`EdgeEvalConfig`, `EdgePolicy`)
- Test: `tests/test_edge_policy_async.py`

**Interfaces:**
- Consumes: `EdgePolicy` (Phase-1: builds base batch from current obs → `extract_actions_hidden_states` → `proj` → `vla_feature`; edge on `(obs96, past96, vla_feature)`), `EdgeEvalConfig`, `resolve_unnorm_key`.
- Produces: `EdgeEvalConfig` gains `base_cadence: int = 1`. `EdgePolicy` gains a step counter and a cache; a new/overridden `act(obs)` that (a) on `self._step % base_cadence == 0` recomputes `vla_feature` from the current frame and stores `self._cached_feature` + `self._cached_base_frame96` (the base's 96px agentview frame), (b) every step runs the edge on `(current 96px frame, self._cached_base_frame96, self._cached_feature)`, (c) increments `self._step`. `reset()` sets `self._step = 0`, clears the cache. Expose `self._step` and cache attrs for the test. `base_cadence=1` reproduces Phase-1 `act` exactly (recompute every step; past frame = the just-processed base frame == current). A helper `_should_refresh(step: int, cadence: int) -> bool` returns `step % cadence == 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_policy_async.py
from experiments.robot.libero.edge_policy import _should_refresh


def test_refresh_cadence():
    # N=1: refresh every step. N=4: steps 0,4,8 refresh; 1,2,3,5 do not.
    assert [ _should_refresh(s, 1) for s in range(5) ] == [True, True, True, True, True]
    assert [ _should_refresh(s, 4) for s in range(8) ] == [True, False, False, False, True, False, False, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_edge_policy_async.py -v`
Expected: FAIL — `_should_refresh` not defined.

- [ ] **Step 3: Implement the async cache**

- Add module-level `def _should_refresh(step: int, cadence: int) -> bool: return step % cadence == 0`.
- Add `base_cadence: int = 1` to `EdgeEvalConfig`.
- In `EdgePolicy.__init__` set `self._step = 0`, `self._cached_feature = None`, `self._cached_base_frame96 = None`, `self.base_cadence = cfg.base_cadence`.
- In `reset()` (add if absent) reset those.
- Refactor `act(obs)` so the base recompute (build base batch → `extract_actions_hidden_states` → `proj.predict_action` → `vla_feature`) and the base's 96px frame capture happen only `if _should_refresh(self._step, self.base_cadence) or self._cached_feature is None`; store into the cache. Then every call runs the edge on `(current_96, self._cached_base_frame96, self._cached_feature)`; unnormalize as before; `self._step += 1`; return the chunk. (Keep the exact frame/normalization helpers Phase-1 `act` uses.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_edge_policy_async.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/robot/libero/edge_policy.py tests/test_edge_policy_async.py
git commit -m "feat: EdgePolicy async cache (base every N steps, edge every step on stale feature)"
```

### Task 4: `--mode async --base_cadence N` in the eval harness

**Files:**
- Modify: `experiments/robot/libero/run_libero_eval.py` (arg parsing + mode dispatch)

**Interfaces:**
- Consumes: `EdgePolicy`/`EdgeEvalConfig` with `base_cadence` (Task 3); the existing `stock`/`edge` mode dispatch and rollout loop.
- Produces: `--mode async` accepted; a `--base_cadence` int arg (default 1) plumbed into `EdgeEvalConfig`. In `async` mode the action source is the same `EdgePolicy` as `edge` mode but with `base_cadence=args.base_cadence`; `policy_reset` calls `EdgePolicy.reset()` each episode. Everything else (env, obs, prompt, unnorm, open-loop cadence, wandb SR logging, per-job seeds) identical to `edge`/`stock`. Run name encodes mode + cadence (e.g. `async-N4-libero_spatial-50trials`).

- [ ] **Step 1: Add the mode + arg**

Add `--base_cadence` (type int, default 1) to the argparser; accept `"async"` in `--mode`. In the mode→policy dispatch, `async` builds the same `EdgePolicy` as `edge` but sets `EdgeEvalConfig.base_cadence = args.base_cadence`. Ensure `policy_reset` invokes `EdgePolicy.reset()` at each episode start so the step counter/cache reset per episode. Include `base_cadence` in the wandb run name/config.

- [ ] **Step 2: Smoke-run one async episode (needs GPU + a Phase-1 checkpoint)**

Run (RTX 5090; ensure GPU clear first — `pgrep -f run_libero_eval` empty):
```bash
MUJOCO_GL=egl .venv/bin/python -m experiments.robot.libero.run_libero_eval --mode async \
  --base_cadence 4 --task_suite_name libero_spatial --num_trials_per_task 1 --num_tasks 1 \
  --edge_ckpt runs/smoke/shead--0_checkpoint.pt --proj_ckpt runs/smoke/proj--0_checkpoint.pt \
  --use_wandb False
```
(Reuse the Task-5.1 fresh/untrained smoke checkpoints, or any Phase-1 checkpoint copied locally.)
Expected: one episode runs end-to-end, valid `[8,7]` chunks each step, base recomputed only every 4th step, prints `success=...` (value irrelevant for a smoke). If it errors, that is a real integration bug — fix before proceeding.

- [ ] **Step 3: Sanity — async N=1 ≡ edge mode**

Run the same command with `--mode edge` and with `--mode async --base_cadence 1` on the SAME checkpoint and a fixed `--seed`, `--num_trials_per_task 2 --num_tasks 1`; confirm identical per-episode success flags (N=1 must reduce to Phase-1 edge behavior). Record the two outputs in the commit message.

- [ ] **Step 4: Commit**

```bash
git add experiments/robot/libero/run_libero_eval.py
git commit -m "feat: --mode async --base_cadence N (cadence-N async LIBERO eval); N=1==edge verified"
```

---

## Milestone C — Latency microbenchmark

### Task 5: `latency_bench.py` — measure `t_base` and `t_edge`

**Files:**
- Create: `experiments/robot/libero/latency_bench.py`
- Test: `tests/test_latency_bench.py`

**Interfaces:**
- Consumes: `build_frozen_base`, `extract_actions_hidden_states`, `Proj_Actiontokens`, `Edge_adapter_manip`, `make_dummy_base_batch`.
- Produces: `benchmark(n_warmup: int = 5, n_iter: int = 50, device=None) -> dict` returning `{"t_base_ms": float, "t_edge_ms": float, "n_iter": int}` where `t_base_ms` = mean wall-clock of one base forward + hidden-state extraction + `proj.predict_action`, `t_edge_ms` = mean of one `Edge_adapter_manip` forward, both with `torch.cuda.synchronize()` around timed regions. A CLI (`python -m ...latency_bench`) prints the dict and the derived per-step compute + speedup for N∈{1,2,4,8,16} (`t_edge + t_base/N`; speedup `t_base/(t_edge+t_base/N)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latency_bench.py
import pytest
pytestmark = pytest.mark.slow


def test_benchmark_returns_positive_times():
    from experiments.robot.libero.latency_bench import benchmark
    out = benchmark(n_warmup=2, n_iter=5)
    assert out["t_base_ms"] > 0.0 and out["t_edge_ms"] > 0.0
    # The edge is ~100x smaller than the 7B base -> must be much faster.
    assert out["t_edge_ms"] < out["t_base_ms"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_latency_bench.py -v -m slow`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the benchmark**

Create `latency_bench.py`: load the frozen base + `make_dummy_base_batch`; build `Proj_Actiontokens(4096,4096,512)` and `Edge_adapter_manip(512)` in fp32 on device; time (with `torch.cuda.synchronize()` and `time.perf_counter()`, `n_warmup` discarded, mean over `n_iter`): (a) `t_base` = `extract_actions_hidden_states(...)` + `proj.predict_action(...)`; (b) `t_edge` = `edge(obs96, past96, vla_feature)` on random 96px tensors. Return the dict. Add a `__main__` that prints `t_base_ms`, `t_edge_ms`, and a small table of per-step compute + speedup for N∈{1,2,4,8,16}.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_latency_bench.py -v -m slow`
Expected: PASS (`t_edge_ms < t_base_ms`, both positive).

- [ ] **Step 5: Commit**

```bash
git add experiments/robot/libero/latency_bench.py tests/test_latency_bench.py
git commit -m "feat: latency_bench for t_base/t_edge + analytic async speedup"
```

---

## Milestone D — Cluster experiments (NCHC Nano4, H200)

> All code from Milestones A–C must be pushed and pulled on the cluster first:
> local `git push exp asyncvla-libero`; on cluster `cd /work/roboleon1295/asyncvla_libero_ws/AsyncVLA-LIBERO && git fetch -q origin && git reset --hard -q origin/asyncvla-libero`.

### Task 6: Delay-aware training run + cadence sweep + latency + results

**Files:**
- Create (cluster): `train_delay.sbatch`, `eval_cadence.sbatch` under the cluster repo root.
- Modify: `docs/superpowers/notes/phase2-results.md` (new; the deliverable write-up).

**Interfaces:**
- Consumes: the pushed Phase-2 code; the Phase-1 sbatch pattern in `nano4-handoff.md`; the Phase-1 edge checkpoint at `runs_libero/asyncvla_libero+b8+lr-0.0001/{shead,proj}--50000_checkpoint.pt`.
- Produces: a delay-aware edge checkpoint; `SR(N)` for both edges across N∈{1,2,4,8,16}; measured `t_base`/`t_edge`; the assembled speedup curve + numbers in `phase2-results.md`.

- [ ] **Step 1: Pilot the delay-aware training (dev partition, short)**

On the cluster, verify the delay-aware CLI runs end-to-end (mirror the Phase-1 pilot):
```bash
srun --account=mst114563 --partition=dev --gres=gpu:1 --cpus-per-task=12 --mem=100G --time=00:15:00 \
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_asyncvla_libero.py \
  --delay_aware True --k_max 15 --max_steps 20 --save_freq 10 --episode_limit 10
```
(with `HF_HOME`/`MUJOCO_GL`/offline wandb env as in Phase 1). Expected: 20 steps run, checkpoint saved, no error. Fix any delay-path runtime bug before the full run.

- [ ] **Step 2: Submit the full delay-aware training (8gpus, ~5h)**

Write `train_delay.sbatch` (copy Phase-1 `train.sbatch`, add `--delay_aware True --k_max 15`, and a distinct `--run_id_note delayaware` so its checkpoints land in a separate `runs_libero/...delayaware/` dir). `sbatch train_delay.sbatch`. Poll `squeue`/wandb until it saves `shead--50000` (read progress from `slurm_logs/`). ~5h on 1×H200.

- [ ] **Step 3: Run the latency microbench (dev partition)**

```bash
srun --account=mst114563 --partition=dev --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=00:20:00 \
  python -m experiments.robot.libero.latency_bench
```
Record `t_base_ms`, `t_edge_ms` and the per-N speedup table into `phase2-results.md`.

- [ ] **Step 4: Cadence sweep — both edges × N∈{1,2,4,8,16} (10 eval jobs)**

Write `eval_cadence.sbatch` parameterized by `--export=ALL,CKPT_DIR=...,STEP=50000,CADENCE=N` running:
`python -m experiments.robot.libero.run_libero_eval --mode async --base_cadence $CADENCE --task_suite_name libero_spatial --num_trials_per_task 50 --edge_ckpt $CKPT_DIR/shead--$STEP_checkpoint.pt --proj_ckpt $CKPT_DIR/proj--$STEP_checkpoint.pt --wandb_project asyncvla-libero`
(with the sim env vars incl. per-job `TMPDIR`). Submit 10 jobs: `CKPT_DIR ∈ {phase1 run, delayaware run}` × `CADENCE ∈ {1,2,4,8,16}`. Serialize/parallelize within the account's job limits; read each `SR(N)` from its `slurm_logs/eval_cadence-*.log` (500 episodes / successes). N=1 for the Phase-1 edge must reproduce ~0.972 (sanity).

- [ ] **Step 5: Assemble the deliverable + commit**

Fill `docs/superpowers/notes/phase2-results.md` with: the two SR-vs-N tables (Phase-1 edge and delay-aware edge), the `t_base`/`t_edge` numbers, the per-step compute + speedup per N, and a short conclusion (does the delay-aware curve stay near 0.972 while the Phase-1 edge degrades?). Include a text/ASCII or matplotlib SR-vs-N plot. Commit and push:
```bash
git add docs/superpowers/notes/phase2-results.md
git commit -m "docs: Phase-2 async speedup results (SR vs cadence, both edges) + latency"
```

---

## Self-Review

**Spec coverage:**
- §4 delay-aware training (base at `I_{t−k}`, edge past = `I_{t−k}`, target at `t`, k~U{0..15}) → Tasks 1, 2. ✔
- §5 cadence-N async eval (cache + hold, refresh every N, N=1≡sync) → Tasks 3, 4. ✔
- §6 metrics: SR(N) both edges → Task 6.4; latency `t_base`/`t_edge` + analytic speedup → Tasks 5, 6.3; deliverable curve → Task 6.5. ✔
- §3 cadence set {1,2,4,8,16}, K_max=15, 500-trial protocol → Global Constraints + Tasks 2/6. ✔
- §7 files: `libero_dataset.py`, `train_asyncvla_libero.py`, `edge_policy.py`, `run_libero_eval.py`, `latency_bench.py`, `phase2-results.md` → Tasks 1–6. ✔
- §8 tests: delay sampler + clamp (T1), delay overfit (T2), async cache cadence + N=1≡edge (T3/T4), async smoke (T4), latency positivity (T5). ✔
- §9 risks: N=1 delay-aware vs Phase-1 comparison is built into the sweep (both edges at every N); analytic-latency labeling in §6/Task 6; cluster footprint via Global Constraints. ✔
- §2 "Phase-1 behavior unchanged" → every task gates new behavior behind a flag; Phase-1 tests re-run in Tasks 1/2. ✔

**Placeholder scan:** External-integration steps (Task 4 smoke, Task 6 cluster runs) use exact commands with expected outputs; code we control (delay sampler, async cache, latency bench) is written in full. Task 1/3 implementation steps say "read the current `__getitem__`/`act` first" because those methods are long Phase-1 code the implementer edits in place — the exact edit points (line ranges, which `t`→`t_base`) are named, not deferred.

**Type consistency:** `delay_aware`/`k_max` identical across dataset (T1), config + `train_one_batch_smoke` (T2). `_sample_delay(t,k_max,rng)->int` (T1) matches its call. `base_cadence` identical across `EdgeEvalConfig`/`EdgePolicy` (T3) and `--base_cadence` (T4). `_should_refresh(step,cadence)->bool` (T3) matches Task-6 semantics. `benchmark(...)->{"t_base_ms","t_edge_ms","n_iter"}` (T5) matches Task 6.3 usage. Item dict keys (`obs_img_96`/`past_img_96`/`gt_action_chunk` + base inputs) unchanged from Phase 1 so `run_forward_pass`/collate need no edits.

**Cross-task note:** Task 6 depends on all of A–C being pushed; Task 4's smoke needs a local checkpoint (reuse the Task-5.1 fresh smoke checkpoints or copy a Phase-1 one). Noted in Task 6 preamble and Task 4 Step 2.
