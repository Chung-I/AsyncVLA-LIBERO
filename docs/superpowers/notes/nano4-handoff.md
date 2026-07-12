# AsyncVLA-LIBERO — Nano4 transition checklist (full training + official eval)

Local Phase-1 (RTX 5090) is code-complete and smoke-validated. The heavy compute — full
edge training and the official 500-trial evals — runs on NCHC Nano4. First read
`~/Codes/console/NCHC_SERVER_README.md` for login, `/work`, transfer, and Slurm guidance.

## What's already done (local, committed on branch `asyncvla-libero`)
- Env with the **OpenVLA-OFT transformers fork** (bidirectional attention) — required, see below.
- All model/data/train/eval code (Milestones 1–5 code), each task-reviewed.
- Stock topline **reproduced ~1.0 (50 trials)** with the fork (vs 0.61 without) → base pipeline validated.
- Edge training overfit test passes; edge eval mode smoke-runs end-to-end.

## What remains (Nano4)
1. **Task 4.3** — full edge+projector training on LIBERO-Spatial (frozen base).
2. **Task 3.2** — official 500-trial **stock** topline → `SR_topline`.
3. **Task 5.2** — official 500-trial **edge** eval with the trained checkpoint → `SR_ours`; report gap.

## Env setup on Nano4 (critical points)
- **Transformers fork is mandatory** regardless of GPU: `moojink/transformers-openvla-oft`
  (transformers 4.40.1) provides the bidirectional attention OpenVLA-OFT parallel decoding
  needs. Without it stock SR collapses (0.61 vs ~1.0). Install:
  `uv pip install --no-deps --reinstall-package transformers "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"`
  then `uv pip install "tokenizers>=0.19.1,<0.20"`.
- **Torch/GPU:** the local torch 2.11.0+cu128 pin was ONLY because the RTX 5090 is Blackwell.
  If Nano4 GPUs are pre-Blackwell (V100/A100/H100), the repo's upstream torch (or a
  cluster-CUDA-matched torch) is fine — but keep the fork + tokenizers<0.20 pins. Verify
  `torch.cuda.is_available()` + a GPU matmul on a Nano4 node.
- Other deps (mirror local env, see `phase1-env.md`): `pip install -e .` with torch pinned
  first so it isn't clobbered; `vint_train` (clone `Learning-to-Drive-Anywhere-with-MBRA`,
  make importable); `efficientnet_pytorch`, `transformers` (fork), LIBERO + robosuite
  (`MUJOCO_GL=egl` for headless), TF/TFDS + h5py, wandb. flash-attn optional (faster training).
- **HF auth:** if `HF_HOME` points at `/work`, `cp ~/.cache/huggingface/token "$HF_HOME/token"`
  first, else downloads stall (project rule).
- **Assets:** download base ckpt `moojink/openvla-7b-oft-finetuned-libero-spatial`; download the
  FULL RLDS `libero_spatial_no_noops` (~1.78 GiB — only 1/16 shard is local). Source noted in
  `phase1-data.md`.

## Run commands (fill X for the node's GPU count / your wandb entity)
```bash
# 4.3 — full edge training (frozen base; edge+proj only)
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_asyncvla_libero.py \
  --wandb_entity <ENTITY> --wandb_project asyncvla-libero --max_steps <N> --batch_size <B>

# 3.2 — official stock topline (500 trials = 10 tasks x 50)
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval \
  --mode stock --task_suite_name libero_spatial --num_trials_per_task 50 --wandb_project asyncvla-libero

# 5.2 — official edge eval with the trained checkpoint (same protocol)
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval \
  --mode edge --task_suite_name libero_spatial --num_trials_per_task 50 \
  --edge_ckpt runs/<run>/shead--<step>_checkpoint.pt --proj_ckpt runs/<run>/proj--<step>_checkpoint.pt \
  --wandb_project asyncvla-libero
```
Report `SR_topline`, `SR_ours`, and the gap (`SR_topline − SR_ours`) — the Phase-1 deliverable.

## Operational cautions (learned locally)
- **One GPU job at a time** — two 7B bases OOM a 32 GB card; on shared nodes, request enough VRAM and serialize.
- Base loads ~14 GB in bf16; training also holds edge+proj + activations.
- The `torch.load` weights_only monkeypatch (LIBERO init states) is scoped in `run_libero_eval.py`.

## Transfer
- Push the branch (or `nano4-xfer` the worktree). Do NOT transfer `.venv`, `runs/`, `wandb/`,
  the checkpoint cache, or `../LIBERO`/`../openvla-oft`/`../Learning-to-Drive-Anywhere-with-MBRA`
  (re-clone/re-provision on the cluster).
