# AsyncVLA-LIBERO **Faithful** Phase 2 — Design Spec

**Date:** 2026-07-13
**Status:** Draft for review
**Author:** Chung-Yi Li (with Claude Code)
**Worktree:** `/home/chungyili/Codes/AsyncVLA-libero-faithful`, branch `asyncvla-libero-faithful`
(forked from `asyncvla-libero` @ `1572c23`)
**Supersedes (for fidelity):** `2026-07-13-asyncvla-libero-phase2-design.md`

## 1. Summary

Our Phase-2 was designed for the *experiment*, not for *fidelity*. An audit against the original
AsyncVLA source found five real deviations. This spec reproduces AsyncVLA's Phase 2 as faithfully
as the embodiment allows, and makes every remaining deviation explicit and justified.

**Deliverable:** the *faithful* async curve — SR vs base cadence `N ∈ {1,2,3,4,6,8}` (500 trials
each) against the validated **0.982** stock topline — plus a **fidelity ledger** marking every
original behavior as *reproduced*, *embodiment-forced*, or *compute-forced*.

**Key context:** the audit confirmed the *core async mechanism* was already faithful — frozen base
+ token projector + edge adapter, trained with a **random delay** so the edge bridges a stale base
feature (`lelan_dataset.py:304`: `lt = random.randint(0, min(iv, 3))`; base and edge-`past` both
read frame `iv-lt`, edge-`obs` reads `iv`), and **closed-loop** execution (edge re-runs every tick,
one action executed). The deviations are capacity, delay range, loss, and augmentation.

## 2. Method: minimal-delta, not copy

This worktree **is a git worktree of the AsyncVLA repo** — the original code is already present and
unmodified (`prismatic/models/small_head.py`, `prismatic/vla/datasets/*`,
`vla-scripts/train_asyncvla.py`, `config_nav/dataset_config.yaml`). There is nothing to copy. The
rule is therefore:

1. **Call the original classes directly.** Use the original **`Edge_adapter`** — not a duplicate.
   It needs exactly **one line**: `nn.Linear(64, 8 * 4)` → `nn.Linear(64, NUM_ACTIONS_CHUNK *
   ACTION_DIM)`. This is a **semantic no-op for navigation** (chunk=8, `ACTION_DIM`=4 → 32,
   identical) and under our LIBERO constants (`ACTION_DIM=7`) it yields 8×7 automatically; the
   reshape already uses `(NUM_ACTIONS_CHUNK, -1)` and adapts for free. **Retire
   `Edge_adapter_manip`** (it was only ever `Edge_adapter` with that output parametrized).
   `Proj_Actiontokens` and `MultiLayerDecoder_trans` are already used unmodified.
2. **Read the original's hyperparameters from the original files** — edge capacity straight from
   `config_nav/dataset_config.yaml` (the same file `train_asyncvla.py` reads).
3. **Add LIBERO-only code** only where the original has *no counterpart* (base loading, eval
   harness, RLDS dataset, gripper convention, center-crop, transformers fork). There is nothing to
   deviate *from* here; this is embodiment-forced infrastructure, already validated to the 0.982
   topline in Phase 1.
4. **Adapt only where embodiment forces it.** That list (§7) *is* the complete deviation ledger.

## 3. The four fidelity fixes

| # | Ours today | Faithful | Original source |
|---|---|---|---|
| 1 | Edge 512/2/2, proj `action_dim=512` | **1024 / 4 heads / 4 layers / ff×4**, proj `action_dim=1024` | `config_nav/dataset_config.yaml` |
| 2 | `k_max = 15` | **`k_max = 3`** — `k = randint(0, min(t, 3))` | `lelan_dataset.py:304` |
| 3 | single **L1** on the chunk | **3-term weighted MSE** (§4) | `train_asyncvla.py:559` |
| 4 | no image augmentation | **random-crop augmentation** (`v_random=0.2`, `h_random=0.1`), shared box | `lelan_dataset.py:113-114, 363-371` |

Already faithful, unchanged: frozen base + train edge & projector only (`TRAIN_BASE=False`,
`TRAIN_HEAD=True`; `train_asyncvla.py:10-11`, trainable set at `:1050-1051`); AdamW @ `1e-4`
(`:209`); chunk = 8; 2× EfficientNet-B0 `from_name` (from scratch); 96×96 edge frames; delayed-frame
training (base at `I_{t−k}`, edge `past` = `I_{t−k}`, target at `t`); closed-loop execution.

## 4. Faithful loss

The original (`train_asyncvla.py:552,557,559`): the edge emits **deltas**; `delta_to_pose`
integrates them into a waypoint trajectory; the loss penalizes both, plus smoothness, plus a
LeLaN-only object-grounding term.

```python
pred      = edge(obs96, past96, vla_feature)            # [B, 8, 7]  (6 EEF deltas + gripper)
pred_traj = torch.cumsum(pred[..., :6], dim=1)          # integrated EEF trajectory  [B, 8, 6]
gt_traj   = torch.cumsum(gt_chunk[..., :6], dim=1)
sm_ref    = torch.cat([torch.zeros_like(pred_traj[:, :1]), pred_traj[:, :-1]], dim=1)

loss = 0.5 * 15.0 * F.mse_loss(pred,      gt_chunk)     # (2) raw deltas         [their 0.5 x 15]
     + 0.5        * F.mse_loss(pred_traj, gt_traj)      # (1) integrated traj    [their 0.5]
     + 0.1        * F.mse_loss(pred_traj, sm_ref)       # (4) smoothness         [their 0.1]
```

**Their exact weights are kept** (`0.5`, `0.5×15 = 7.5`, `0.1`). The gripper participates **only** in
the delta term — it is an absolute command and does not integrate. `sm_ref` mirrors
`cat(action_orig, predicted_actions[:, 0:-1])`: the "previous predicted waypoint", with the chunk's
first entry compared against the origin (the current EEF pose is the origin of the relative frame).

The original masks the delta/traj terms to `~lan_bool` (non-LeLaN samples); with a single dataset
there is no mask.

**Dropped:** their 4th term `0.1 * MSE(obj_pose_norm, predicted_actions[:, -1, 0:2])` — LeLaN
language-object grounding, no LIBERO analog (§7).

## 5. Faithful training configuration

Frozen OpenVLA-OFT LIBERO base; train **`Edge_adapter`(1024, 4, 4, ff=4)** + **`Proj_Actiontokens`
(`action_dim=1024`)**; `k_max = 3`; 3-term MSE; random-crop augmentation; AdamW @ `1e-4`;
**50k steps, batch 8, 1×H200**; checkpoints every 5k; W&B.

**Random-crop port** (`lelan_dataset.py:363-371`), applied at **training only**:
```python
voffset = int(H * 0.2 * random.random())      # v_random = 0.2
hoffset = int(W * 0.1 * random.random())      # h_random = 0.1
box     = (hoffset, voffset, W - hoffset, H - voffset)
```
**One box per sample**, applied to *all four* images — the base's agentview, the base's wrist, and
both 96px edge frames (current `I_t` and delayed `I_{t−k}`) — so the edge's current-vs-base
comparison stays consistent, exactly as the original shares one `PILbox` across its base/current/goal
frames. The crop happens on the raw frame, **before** the existing base transform
(`prepare_images_for_vla`, center-crop) and the 96px edge transform. Offsets are expressed as
**fractions** of the frame (0–20% vertical, 0–10% horizontal), preserving the original's ratios on
our 256px frames.

*On training scale:* the original ran 750k steps on 5×H200 over a multi-dataset navigation corpus.
That is neither reproducible on our budget (≈77 h on one GPU) nor meaningful on a ~25× smaller
dataset, and Phase-1 evidence shows the edge **saturates by ~10k steps** (0.966 @10k → 0.972 @50k).
Treated as **compute-forced**, documented, not claimed as fidelity.

## 6. Async evaluation

**The existing harness is reused unchanged** — it is already faithful, verified against
`inference/run_asyncvla.py` (`tick_rate = 3`; `pd_controller` selects a single waypoint and
re-plans every tick) and paper Algorithm 1:

- **Closed-loop**: the edge runs **every env step** and executes `actions[0]` (receding horizon).
- The base recomputes every **N env steps**; the edge's `past` frame is the **base's cached frame**.
- Staleness `k = t − t' ∈ {0..N−1}` in **frame units** — the same units `k_max` is sampled in.

**Sweep `N ∈ {1, 2, 3, 4, 6, 8}`**, 500 trials each (10 tasks × 50), same protocol as the topline.
With `k_max = 3`, **N ≤ 4 is in-distribution** (the regime the original designed for: staleness ≤ 3
ticks ⇒ base every ≤ 4 ticks ⇒ `f_s/f_b ≈ 4`). **N = 6, 8 are out-of-distribution** and are included
to expose the cliff — itself a finding about what `k_max` buys.

## 7. Deviation ledger (the complete list)

**Embodiment-forced**

| Deviation | Why |
|---|---|
| `delta_to_pose` → **cumsum** over the 6 EEF dims | Original (`train_asyncvla.py:281`) is **SE(2) body-frame** composition — each delta is rotated by the accumulated heading (`dx_w = cosθ·dx − sinθ·dy`), θ accumulates. LIBERO OSC deltas are **world-frame**, so they add: cumsum is the correct analog (exact for translation; small-angle approximation for axis-angle rotation). |
| Drop `obj_pose` loss term | LeLaN language-object grounding; no LIBERO analog. |
| Drop horizontal-flip augmentation (keep random crop) | Their flip is valid *only because they mirror the actions* (`nomad_traj_norm[:,1] = -...`, heading `sin`). Mirroring 6-DoF EEF + gripper (negate y, flip rotations about x/z, asymmetric wrist view) is error-prone. |
| Base model: OmniVLA 8.27B → OpenVLA-OFT 7B LIBERO | Different embodiment/task. |
| Action space: 2D waypoints + `delta_to_pose` + PD controller → 7-DoF EEF deltas + gripper | Embodiment. |
| "Proprio" slot carries the **goal pose** (nav goal conditioning) → **robot state** (EEF + gripper) | Manipulation needs proprioception; navigation needs a goal. |
| 9 modality IDs (satellite / pose / image / language) → **1** (`taskid = 0`) | LIBERO is language-only. |
| `L1RegressionDistHead` (distance head) | Navigation-only. |
| Datasets (LeLaN / GNM / SACSoN) → LIBERO-Spatial RLDS | Embodiment. |
| Nav reuses the 2nd image slot for the **goal image**; we use a real **wrist** camera | Embodiment. |
| 3 Hz real-robot control, PD `DT = 1/3` → LIBERO sim step rate | Embodiment. |
| Real-robot eval (Vizbot / Jetson Orin) → LIBERO sim success rate | Embodiment. |

**Compute-forced**

| Deviation | Why |
|---|---|
| 50k steps / batch 8 / 1×H200 (vs 750k / 5×H200 / grad-accum 2) | ≈77 h on one GPU; ~25× smaller dataset; edge saturates by ~10k steps. |
| LR decay 10× at 100k steps (`train_asyncvla.py:211`) never triggers | We train 50k. Moot, not removed. |

## 8. Files (this worktree)

- `prismatic/models/small_head.py` — **one line**: `Edge_adapter`'s `nn.Linear(64, 8 * 4)` →
  `nn.Linear(64, NUM_ACTIONS_CHUNK * ACTION_DIM)`. Retire `Edge_adapter_manip`.
- `experiments/robot/libero/edge_arch.py` — build the original **`Edge_adapter`**; defaults read the
  faithful values from `config_nav/dataset_config.yaml` (1024/4/4/4).
- `prismatic/vla/datasets/libero_dataset.py` — random-crop augmentation (shared box, train-only);
  `k_max` default **3**.
- `vla-scripts/train_asyncvla_libero.py` — the 3-term MSE loss (§4); `k_max` default 3; `--image_aug`.
- `experiments/robot/libero/run_libero_eval.py`, `edge_policy.py`, `latency_bench.py` — **unchanged**
  (already faithful; `latency_bench` picks up the new capacity via `EdgeArch`).
- `docs/superpowers/notes/faithful-results.md` — the deliverable write-up.

## 9. Testing

- **Unit:** `Edge_adapter` at `config_nav` values (1024/4/4/4) outputs `[B, 8, 7]`; the one-line
  change is a no-op at `ACTION_DIM=4` (assert the nav shape is unchanged).
- **Unit:** delay sampler with `k_max=3` — `k ∈ {0..3}`, clamped at episode start.
- **Unit:** random crop — **one box per sample** shared across base/wrist/both edge frames; the crop
  is **off** at eval.
- **Unit:** the 3-term loss — each term's weight; the gripper is excluded from the traj/smoothness
  terms; gradients reach edge + projector only.
- **Integration:** delay-aware overfit at the faithful config (loss drops well below half).
- **Smoke:** 1 async episode at `N = 3` (in-distribution), end-to-end, valid `[8,7]` chunks.

## 10. Risks

1. **The 3-term MSE may train differently than L1.** The `7.5 ×` delta weight is large; if training is
   unstable, that is a *finding* about the original's recipe under a different embodiment — report it,
   do not silently retune. (Their weights are kept exactly, per decision.)
2. **`k_max = 3` caps usable cadence at ~4.** Expected — and the N=6,8 points are there to show it.
   This is the faithful result, not a failure.
3. **Random-crop interacts with the base's center-crop.** Train-time random crop + eval-time
   center-crop is standard (and what OpenVLA-OFT does), but the base was validated *without* train-time
   aug; if the topline-relative SR drops, suspect the crop first.

## 11. Out of scope

- Joint base+edge training (`TRAIN_BASE=True` + LoRA) — the paper's headline, but the released
  default is `TRAIN_BASE=False`; decided out of scope.
- Threaded / real-wall-clock true-async runtime; robustness-to-perturbation study; LIBERO suites
  beyond Spatial; the `k_max=3` vs `k_max=15` controlled ablation (a possible follow-up).
