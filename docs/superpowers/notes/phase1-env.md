# Phase 1 environment — AsyncVLA-LIBERO (Task 0.1 / Milestone 0)

Development + smoke-test box: local workstation with an **RTX 5090 (Blackwell, sm_120)**,
driver 580.159.03, CUDA 13.0. Full training and the 500-trial eval run later on NCHC Nano4.

Worktree: `/home/chungyili/Codes/AsyncVLA-libero` (branch `asyncvla-libero`).

## Environment manager & interpreter

- No conda, no pre-existing venv. System `python3` is 3.12 with no pip — not used.
- Tool: **uv 0.11.19** at `~/.local/bin/uv` (add `~/.local/bin` to `PATH`).
- venv: `uv venv --python 3.10 .venv` → **CPython 3.10.20** at
  `/home/chungyili/Codes/AsyncVLA-libero/.venv`. Activate with `source .venv/bin/activate`
  before every install / run. `.venv/` is gitignored.

## GPU / torch (Blackwell sm_120)

The upstream pins `torch==2.2.0` etc. CANNOT drive Blackwell. Installed the cu128 channel instead:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Resolved (all `+cu128`):
- **torch 2.11.0+cu128**
- **torchvision 0.26.0+cu128**
- **torchaudio 2.11.0+cu128** — see gotcha below.

Verified on the 5090:
```
torch 2.11.0+cu128  cuda.is_available()=True  device="NVIDIA GeForce RTX 5090"
(x@x).sum() -> finite number, no CUDA kernel error   # G1 PASS
```

### torchaudio gotcha
`pip install -e .` (and any later resolve) pulls plain `torchaudio==2.11.0` from PyPI, which
links against **libcudart.so.13** (CUDA 13) not present in the venv → `OSError: libcudart.so.13`.
`torchaudio` is NOT used by the VLA/edge/eval import path, but the pin forces it in. Fix — force
the cu128 build:
```bash
uv pip install --reinstall-package torchaudio torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```
(Plain `uv pip install torchaudio --index-url .../cu128` is a no-op: uv treats 2.11.0 as already
satisfying the requirement. Use `--reinstall-package`.)

## Installing the project (`prismatic` / package name `OmniVLA`)

`pip install -e .` would drag in `torch==2.2.0` and destroy the Blackwell torch. Fix: **relaxed the
pins in `pyproject.toml`** (committed on this branch) so the editable install keeps the Blackwell
stack. Changes made:
- `torch>=2.2.0`, `torchvision>=0.17.0`, `torchaudio>=2.2.0` (was `==` pins).
- `tokenizers>=0.19.1` (was `==0.19.1`).
- `transformers @ git+moojink/transformers-openvla-oft.git` → **`transformers>=4.40.1,<4.48`**.
  The git fork is pinned to old torch/tokenizers internals and did not resolve against torch 2.11;
  resolved to a released **transformers 4.47.1** (compatible with OpenVLA-OFT remote code).
  CAVEAT: the fork existed for bidirectional attn (parallel decoding); if parallel-decoding
  numerics differ later, revisit this pin.
- `tensorflow==2.15.0` → `tensorflow>=2.15.0` (resolved **2.21.0**).
- `tensorflow_datasets==4.9.3` → `>=4.9.3` (resolved **4.9.10**).
- Dropped `tensorflow_graphics==2021.12.3` and `dlimp @ git...` from the pin list, installed them
  separately with `--no-deps` (see below).
- `diffusers==0.11.1` **kept pinned** — it's only in a commented-out import
  (`prismatic/models/action_heads.py`); newer diffusers hard-require `peft>=0.15`, which conflicts
  with the OpenVLA-OFT `peft==0.11.1` pin. `uv pip install -e .` first resolved diffusers 0.34.0
  (broke `import diffusers`); re-pinned to 0.11.1.

Install:
```bash
uv pip install -e .        # keeps torch 2.11.0+cu128 intact
```

### dlimp — needed for `import prismatic`
`import prismatic` transitively imports the RLDS dataset chain → `import dlimp`. Install separately
(without deps, so it can't clobber torch/tf):
```bash
uv pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```
Resolved: **dlimp 0.0.1** (commit 040105d).

### tensorflow-metadata / protobuf conflict
`import prismatic` → tensorflow_datasets → `tensorflow-metadata`, which `pip install -e .` resolved
to the ancient **0.5.0**, whose generated `_pb2` files are incompatible with protobuf 7
(`TypeError: Descriptors cannot be created directly`). Fix:
```bash
uv pip install --upgrade "tensorflow-metadata>=1.16.0"   # -> 1.21.0, pulls protobuf 6.32.0
```

### tensorflow-graphics — needed for `import prismatic`
The RLDS OXE config imports `tensorflow_graphics`. The upstream 2021.12.3 pin has no clean
dependency resolution on this env; install without deps (it imports fine against TF 2.21):
```bash
uv pip install --no-deps tensorflow-graphics    # -> 2021.12.3, imports OK
```

## vint_train (REQUIRED — brief omitted it)

`prismatic/models/small_head.py` imports
`from vint_train.models.vint.self_attention import MultiLayerDecoder_trans` at module top, so the
Edge Adapter can't import without it. Its code lives under `train/` in the MBRA repo. Cloned as a
sibling and made importable via a `.pth` file (self_attention.py only needs torch — no heavy MBRA
deps pulled):

```bash
git clone --depth 1 https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA \
    /home/chungyili/Codes/Learning-to-Drive-Anywhere-with-MBRA
# add a path file into the venv site-packages pointing at the train/ dir:
echo "/home/chungyili/Codes/Learning-to-Drive-Anywhere-with-MBRA/train" \
    > .venv/lib/python3.10/site-packages/vint_train.pth
```
Verify: `python -c "from vint_train.models.vint.self_attention import MultiLayerDecoder_trans; print('vint_train ok')"`.

## flash-attn — SKIPPED on this box (by design)

`experiments/robot/openvla_utils.py::get_vla` has `attn_implementation` commented out, so the base
loads without flash-attn (eager/sdpa). flash-attn is only needed for fast *training*, which happens
on NCHC Nano4 later. Not built here — do not burn time on it.

## Other deps

Installed into the venv:
- `efficientnet-pytorch 0.7.1`, `transformers 4.47.1`, `tokenizers 0.21.4`, `timm 0.9.10`,
  `wandb 0.28.0`, `pytest 9.1.1`, `imageio 2.37.3`, `imageio-ffmpeg 0.6.0`.
- `tensorflow 2.21.0`, `tensorflow-datasets 4.9.10` (RLDS, used later).
- `accelerate 1.14.0`, `peft 0.11.1`, `diffusers 0.11.1`, `huggingface-hub 0.29.1`.
- **numpy 1.26.4** — was 2.2.6 until the LIBERO/robosuite stack (numba) forced it to <2. torch 2.11
  and TF 2.21 both work with numpy 1.26.4 (re-verified all gates after the downgrade).

## LIBERO + robosuite + mujoco (sim) — WORKING (exceeds best-effort target)

A LIBERO clone already existed as a sibling at `/home/chungyili/Codes/LIBERO` (reused, not
re-cloned). Its `setup.py` has `install_requires=[]`, so editable install pulls nothing:
```bash
uv pip install -e /home/chungyili/Codes/LIBERO
```
`from libero.libero import benchmark` works and lists `libero_spatial` (that alone satisfies G5).

For actual headless sim rollouts the runtime deps had to be pinned carefully:
- **robosuite 1.4.0** (LIBERO's pin). `robosuite` (latest, 1.5.2) restructured modules
  (`robosuite.environments.manipulation.single_arm_env` gone) → LIBERO import fails. Must be 1.4.0.
- **mujoco 2.3.2**. robosuite 1.4.0 declares only `mujoco>=2.3.0`; the resolver first installed
  mujoco 3.10.0, which changed the `mj_fullM()` signature → `TypeError` at `env.step()`. Pinning
  mujoco 2.3.2 fixed it.
- Plus: `bddl 3.6.0`, `easydict 1.13`, `hydra-core 1.3.4`, `cloudpickle 3.1.2`, `gym 0.26.2`,
  `thop`, `robomimic 0.3.0`.

Headless render requires **`MUJOCO_GL=egl`**. Verified end-to-end:
```bash
MUJOCO_GL=egl python -c "... OffScreenRenderEnv(libero_spatial task0) ...; env.reset(); env.step([0]*7)"
# -> SIM_OK agentview_image (128, 128, 3)
```

## Base checkpoint (G4)

`HF_HOME` is unset, so the token at `~/.cache/huggingface/token` is used (present) — no copy needed.
```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial'))"
```
Local path:
`/home/chungyili/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b`

## Verification gates (all PASS)

| Gate | Check | Result |
|------|-------|--------|
| G1 | torch imports, CUDA available, 5090 matmul | PASS (torch 2.11.0+cu128, "NVIDIA GeForce RTX 5090", finite matmul) |
| G2 | `import prismatic; from prismatic.vla.constants import ACTION_DIM` | PASS (ACTION_DIM=4) |
| G3 | `from prismatic.models.small_head import Edge_adapter` | PASS (vint_train + efficientnet wired) |
| G4 | base checkpoint downloaded | PASS (path above, ~14GB) |
| G5 | LIBERO `libero_spatial` importable (+ full sim verified) | PASS |

Also verified: `import experiments.robot.openvla_utils` works (brief's stated interface goal).

## Reproduce from scratch (summary)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/chungyili/Codes/AsyncVLA-libero
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
uv pip install --reinstall-package torchaudio torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
uv pip install --upgrade "tensorflow-metadata>=1.16.0"
uv pip install --no-deps tensorflow-graphics
uv pip install "diffusers==0.11.1"
uv pip install pytest imageio-ffmpeg
git clone --depth 1 https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA ../Learning-to-Drive-Anywhere-with-MBRA
echo "$(cd ../Learning-to-Drive-Anywhere-with-MBRA/train && pwd)" > .venv/lib/python3.10/site-packages/vint_train.pth
# LIBERO sim (best-effort, working):
uv pip install -e ../LIBERO   # or /home/chungyili/Codes/LIBERO
uv pip install "robosuite==1.4.0" "mujoco==2.3.2" bddl easydict "hydra-core>=1.2" cloudpickle "gym>=0.25,<0.27" thop robomimic
# checkpoint:
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial'))"
```

## UPDATE (2026-07-12): restored OpenVLA-OFT transformers fork (bidirectional attention)

**Why:** Stock `transformers==4.47.1` uses causal attention; OpenVLA-OFT parallel decoding was
trained with BIDIRECTIONAL attention over action tokens (implemented by the fork). Symptom: stock
LIBERO-Spatial topline SR = **0.61** (100 trials) vs published ~0.9+. Root cause documented in the
checkpoint's own `modeling_prismatic.py:742` ("non-causal bi-directional self-attention").

**Fix (keeps torch 2.11.0+cu128 / Blackwell):**
```
uv pip install --python .venv/bin/python --no-deps --reinstall-package transformers \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
uv pip install --python .venv/bin/python "tokenizers>=0.19.1,<0.20"
```
Result: transformers 4.40.1 (fork), tokenizers 0.19.1, torch 2.11.0+cu128 intact. Base loads with
NO version-mismatch warning. SR recovery confirmed separately (see phase1-results.md).
