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

This worktree **is a git worktree of the AsyncVLA repo**, and most of the original code is present
and unmodified. But `git diff main --stat` shows three original files ARE modified on this branch:
`prismatic/vla/constants.py` (the LIBERO retarget, commit `7954582`), `prismatic/models/action_heads.py`
(a new `L1RegressionActionHead` class added, additive), and `prismatic/models/small_head.py` (the
`action_dim` parameterization described below). §7's deviation ledger records the consequences.
The rule for everything else is therefore:

1. **Call the original classes directly.** Use the original **`Edge_adapter`** — not a duplicate.
   Its output head width becomes a constructor parameter: `nn.Linear(64, 8 * 4)` →
   `nn.Linear(64, NUM_ACTIONS_CHUNK * action_dim)`, with `action_dim: int = 4` defaulting to the
   original nav width so every existing original caller (`train_asyncvla.py`,
   `run_asyncvla.py`, neither of which ever passes `action_dim`) is byte-identical to before —
   `NUM_ACTIONS_CHUNK * 4 == 8 * 4 == 32`. **This is NOT a no-op under `prismatic/vla/constants.py`'s
   `ACTION_DIM`** — upstream AsyncVLA (`main`) defines `ACTION_DIM = 4` (and `POSE_DIM = 4`), the
   nav action width; the `asyncvla-libero` branch this branch forked from retargeted both to `7`
   for LIBERO's 7-DoF action space (commit `7954582`), so on THIS branch `ACTION_DIM` is `7` and no
   longer means "the nav width". The head width must therefore NOT be derived from `ACTION_DIM`,
   because this branch has repurposed that constant: hardcoding `NUM_ACTIONS_CHUNK * ACTION_DIM`
   would silently build a 56-wide head for the original nav callers too and break both
   `train_asyncvla.py`'s loss (shape mismatch against the 4-D `daction_ref`) and
   `run_asyncvla.py`'s checkpoint load (`[32,64]` vs `[56,64]`). Hardcoding the nav default `4` in
   the signature instead keeps `Edge_adapter()` correct for the original's callers *independently*
   of the constants retarget, while the LIBERO path passes `action_dim=ACTION_DIM` (=7)
   **explicitly**, via `build_edge_and_proj`. The reshape at the
   end of `forward` already uses `(NUM_ACTIONS_CHUNK, -1)` and adapts for free either way.
   **Retire `Edge_adapter_manip`** (it was only ever `Edge_adapter` with that output
   parametrized). `Proj_Actiontokens` and `MultiLayerDecoder_trans` are already used unmodified.
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
| 3 | single **L1** on the chunk | **2-term weighted MSE** (§4) — their delta + trajectory terms; their 3rd ("smoothness") term is DELIBERATELY DROPPED, not reproduced (§7) | `train_asyncvla.py:552,557,559` |
| 4 | no image augmentation | **random-crop augmentation** (`v_random=0.2`, `h_random=0.1`), shared box | `lelan_dataset.py:113-114, 363-371` |

Already faithful, unchanged: frozen base + train edge & projector only (`TRAIN_BASE=False`,
`TRAIN_HEAD=True`; `train_asyncvla.py:10-11`, trainable set at `:1050-1051`); AdamW @ `1e-4`
(`:209`); chunk = 8; 2× EfficientNet-B0 `from_name` (from scratch); 96×96 edge frames; delayed-frame
training (base at `I_{t−k}`, edge `past` = `I_{t−k}`, target at `t`); closed-loop execution.

## 4. Faithful loss

The original (`train_asyncvla.py:552,557,559`): the edge emits **deltas**; `delta_to_pose`
integrates them into a waypoint trajectory; the loss penalizes both, plus a "smoothness" term,
plus a LeLaN-only object-grounding term.

```python
pred      = edge(obs96, past96, vla_feature)            # [B, 8, 7]  (6 EEF deltas + gripper)
pred_traj = torch.cumsum(pred[..., :6], dim=1)          # integrated EEF trajectory  [B, 8, 6]
gt_traj   = torch.cumsum(gt_chunk[..., :6], dim=1)

loss = 0.5 * 15.0 * F.mse_loss(pred,      gt_chunk)     # (2) raw deltas         [their 0.5 x 15]
     + 0.5        * F.mse_loss(pred_traj, gt_traj)      # (1) integrated traj    [their 0.5]
```

**Their exact weights are kept** on the two remaining terms (`0.5`, `0.5×15 = 7.5`) — NOT
renormalized after dropping the 3rd term. The gripper participates **only** in the delta term —
it is an absolute command and does not integrate.

The original masks the delta/traj terms to `~lan_bool` (non-LeLaN samples); with a single dataset
there is no mask.

**Dropped, not reproduced:**
- Their 3rd term, `0.1 * MSE(sm_ref, predicted_actions)` ("smoothness") — proved to be a
  mislabeled bug, not a fidelity gap that can be ported. See §7 for the closed-form derivation
  and the decision to drop rather than reproduce it.
- Their 4th term `0.1 * MSE(obj_pose_norm, predicted_actions[:, -1, 0:2])` — LeLaN
  language-object grounding, no LIBERO analog (§7).

## 5. Faithful training configuration

Frozen OpenVLA-OFT LIBERO base; train **`Edge_adapter`(1024, 4, 4, ff=4)** + **`Proj_Actiontokens`
(`action_dim=1024`)**; `k_max = 3`; 2-term MSE (§4, §7 — smoothness term dropped); random-crop augmentation; AdamW @ `1e-4`; `MultiStepLR` decay (§7);
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
| **Trajectory-term rotation approximation** | `cumsum` treats the 3 rotation dims (axis-angle) of the LIBERO action as a vector space, i.e. integrates them by addition. 3-D rotations don't commute in general, so this is a first-order (small-angle) approximation, not exact composition (unlike the 2-D `cosθ/sinθ` heading case above, which is exact for the translation part and only approximate for the heading itself). It holds well in practice: LIBERO's per-step rotation deltas are small — `bounds_q99` span ≈0.21–0.38 rad for the 3 rotation dims vs ≈1.5–1.9 for the 3 translation dims — so consecutive small rotations approximately commute over an 8-step chunk. |
| **Smoothness term DROPPED — reproduces a bug, not a fidelity gap** (supersedes the earlier "Smoothness-term target no longer means 'no motion'" framing, which treated this as merely embodiment-forced; we now drop the term outright rather than adapt it) | Their term is `0.1 * MSE(sm_ref, predicted_actions)`, where `predicted_actions = delta_to_pose(deltas)` is SE(2) composition and `sm_ref` is the previous predicted pose. Writing the pose as `p_t = (x, y, cosθ, sinθ)`, the consecutive-pose difference has xy-part `R(θ_{t−1})·d_t` and heading-part with squared norm `2 − 2cos(Δθ_t)`. Because a rotation preserves norm, `‖R·d‖² = ‖d‖²`, so the whole term collapses in closed form to `nav_smooth = ¼ · mean_t[dx_t² + dy_t² + 4·sin²(Δθ_t/2)]` (the ¼ comes from `nn.MSELoss` averaging over all 4 pose channels x, y, cosθ, sinθ) — a pure function of the predicted DELTAS' MAGNITUDES, with no ground truth and no curvature term. It is "take small steps," NOT "don't jerk" — a mislabeled magnitude penalty. (Verified numerically against their verbatim `delta_to_pose`, exact match to the ¼ form: 0.53007358 vs 0.53007358.) Under our LIBERO `bounds_q99` OFFSET normalization it is worse than useless: normalized-zero is the MIDPOINT of `[q01, q99]`, not zero motion, so the term pulls the policy toward a constant raw drift (≈ +0.096/+0.107 in x/y) rather than toward stillness or smoothness. We drop it rather than reproduce a proven bug. |
| **Loss weight ROLES are reversed** | In the original, WAYPOINTS are the primary supervised data (`nomad_traj_norm`, "normalized pose on robot coordinate") and the deltas are DERIVED via `pose_to_delta`; in LIBERO the DELTAS are primary (the OSC commands in the RLDS `action` field) and the trajectory is derived via `cumsum`. We kept their numeric weights (15× on delta, 0.5 on trajectory), but those weights now sit on the opposite quantities — numerically identical weights, semantically different objective. |
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

**Unforced (deliberate)**

| Deviation | Why |
|---|---|
| **`MultiStepLR` restored, milestone rescaled** | Neither LIBERO branch had ANY scheduler (flat `1e-4` throughout); the original decays the LR 10× partway through training (`train_asyncvla.py:1059-1063`: `MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)`, with `num_steps_before_decay=100_000` of `max_steps=200_000` — i.e. at the HALFWAY point). We restore the scheduler and set `num_steps_before_decay=25_000` — half of OUR `max_steps=50_000` — preserving their 50% ratio rather than their literal `100_000` value, which would never trigger at our shorter training length. Their optional warmup (`lr_warmup_steps`, default 0) is unused in the original and is skipped. |
| `shead`/`action_proj` trained in **fp32** → original trains in **bf16** | Original (`train_asyncvla.py:1022,1031`) casts `shead`/`action_proj` to bf16. We build/train both in fp32 (`edge_arch.py:103-104`). Not embodiment-forced — a deliberate choice (numerical headroom for a from-scratch small head on ~25x less data), recorded here so it isn't mistaken for an oversight. |
| Original nav code paths no longer runnable on this branch | The inherited LIBERO retarget of `prismatic/vla/constants.py` (`ACTION_DIM`/`POSE_DIM` 4 → 7, commit `7954582`) means the ORIGINAL nav code paths (`vla-scripts/train_asyncvla.py`, `inference/run_asyncvla.py`) are no longer runnable on this branch: `Proj_Actiontokens` (`prismatic/models/small_head.py:233`) builds `MLPResNet_idcat(input_dim=4096*7=28672)` where the released nav `action_proj` checkpoint expects `4096*4=16384`. `Edge_adapter`'s own `shead` checkpoint DOES still load (its head defaults to the nav width 4 → 32). This is accepted: the fork is LIBERO-only. It is recorded here because the minimal-delta method's premise is that the original code is present and working, and for the nav path that is now only partly true. |

## 8. Files (this worktree)

- `prismatic/models/small_head.py` — `Edge_adapter` gains an `action_dim: int = 4` constructor
  kwarg (4 = the original nav width, so every original caller is unchanged);
  `nn.Linear(64, 8 * 4)` → `nn.Linear(64, NUM_ACTIONS_CHUNK * action_dim)`. Retire
  `Edge_adapter_manip`.
- `experiments/robot/libero/edge_arch.py` — build the original **`Edge_adapter`**, passing
  `action_dim=ACTION_DIM` (=7) explicitly for LIBERO; capacity (1024/4/4/4) read from
  `config_nav/dataset_config.yaml`. `build_edge_and_proj`'s `arch` argument is required (no
  default) so no caller can silently fall back to the unfaithful 512/2/2/4 capacity.
- `prismatic/vla/datasets/libero_dataset.py` — random-crop augmentation (shared box, train-only);
  `k_max` default **3**.
- `vla-scripts/train_asyncvla_libero.py` — the 2-term MSE loss (§4, §7); `k_max` default 3; `--image_aug`; `MultiStepLR` decay (§7).
- `experiments/robot/libero/latency_bench.py` — **unchanged** (already faithful; picks up the
  new capacity via `EdgeArch`).
- `experiments/robot/libero/edge_policy.py` — `EdgePolicy.__init__` now resolves its edge
  architecture via `edge_arch.load_edge_arch` (raises if `edge_arch.json` is missing and no
  explicit `EdgeArch` is passed) and cross-checks the loaded checkpoint against it via
  `edge_arch.validate_edge_arch` before `load_state_dict`. It also accepts an optional
  `edge_arch: Optional[EdgeArch] = None` constructor arg — the escape hatch for checkpoints
  saved before `edge_arch.json` existed (e.g. historical Phase-1 checkpoints), passed through
  to `load_edge_arch(edge_ckpt, arch=edge_arch)`.
- `experiments/robot/libero/run_libero_eval.py` — exposes that escape hatch as four CLI flags
  (`--edge_arch_obs_encoding_size`, `--edge_arch_mha_num_attention_heads`,
  `--edge_arch_mha_num_attention_layers`, `--edge_arch_mha_ff_dim_factor`; must be given all
  four together or not at all) so a checkpoint predating `edge_arch.json` can still be
  evaluated under `--mode edge`/`--mode async` without a code change. **Every LIBERO
  checkpoint saved by `train_asyncvla_libero.py` (this branch) already writes
  `edge_arch.json` next to it (§7/§9), so this flag is only needed for a checkpoint that
  predates this branch — the default (no flags) still raises loudly rather than guessing.**
- `docs/superpowers/notes/faithful-results.md` — the deliverable write-up.

## 9. Testing

- **Unit:** `Edge_adapter` at `config_nav` values (1024/4/4/4) with `action_dim=ACTION_DIM` outputs
  `[B, 8, 7]`; the default (no `action_dim` passed) stays `8 * 4 == 32`-wide (assert BOTH widths).
- **Unit:** delay sampler with `k_max=3` — `k ∈ {0..3}`, clamped at episode start.
- **Unit:** random crop — **one box per sample** shared across base/wrist/both edge frames; the crop
  is **off** at eval.
- **Unit:** the 2-term loss — each term's weight; the gripper is excluded from the traj term;
  gradients reach edge + projector only.
- **Unit:** `MultiStepLR` — LR is `1e-4` before `cfg.num_steps_before_decay` and `1e-5` after.
- **Integration:** delay-aware overfit at the faithful config (loss drops well below half).
- **Smoke:** 1 async episode at `N = 3` (in-distribution), end-to-end, valid `[8,7]` chunks.

## 10. Risks

1. **The 2-term MSE may train differently than L1.** The `7.5 ×` delta weight is large; if training is
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
