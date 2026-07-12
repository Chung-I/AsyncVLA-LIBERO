# AsyncVLA-LIBERO Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully port AsyncVLA's frozen-base + trainable Edge-Adapter structure to LIBERO-Spatial manipulation, and report our base+edge success rate (SR) against the stock OpenVLA-OFT-LIBERO topline SR in the same simulator.

**Architecture:** A frozen OpenVLA-OFT LIBERO base (loaded via the repo's existing `experiments/robot/openvla_utils.py::get_vla`) emits action-token hidden states → a trainable `Proj_Actiontokens` projector compresses them to a `vla_feature` → a trainable EfficientNet-B0 `Edge_adapter` fuses `vla_feature` with the current + previous camera frame to predict an 8×7 action chunk. Evaluation runs two modes in the LIBERO sim under one protocol: `stock` (base's native head → `SR_topline`) and `edge` (our base+edge → `SR_ours`). No two-rate async loop / latency study (that is Phase 2).

**Tech Stack:** PyTorch, HuggingFace Transformers (OpenVLA-OFT remote code), Prismatic VLA utilities (vendored), EfficientNet-PyTorch, LIBERO + robosuite simulator, RLDS / TensorFlow-Datasets for demo data, Weights & Biases, DDP/`torchrun`.

## Global Constraints

- Work only in the worktree `/home/chungyili/Codes/AsyncVLA-libero` on branch `asyncvla-libero`. Leave `main` and the navigation code paths untouched except where a task explicitly edits a shared module.
- Do NOT read or copy from `/home/chungyili/Codes/async_pi0`.
- Base VLA is **frozen** in all Phase-1 code: load under `torch.no_grad()` / `requires_grad_(False)`; only the Edge Adapter + `Proj_Actiontokens` projector are trained. No LoRA in Phase 1.
- Base checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial` (HF Hub). Its companion `proprio_projector--150000_checkpoint.pt`, FiLM, and `dataset_statistics.json` are loaded by `openvla_utils.py`.
- LIBERO action space: `ACTION_DIM = 7` (6 EEF deltas + 1 gripper), `NUM_ACTIONS_CHUNK = 8`.
- Edge encoders: EfficientNet-B0 initialized **from scratch** (`EfficientNet.from_name`, never `.from_pretrained`).
- Eval protocol (both modes, identical): LIBERO-Spatial = 10 tasks × 50 rollouts = 500 trials, same seeds/config.
- All training and eval runs log to Weights & Biases (project rule).
- Edge frame size = 96×96, normalized with ImageNet mean/std (mirror `inference/run_asyncvla.py`'s `transform`). Base image size/preprocessing is whatever `openvla_utils`/the processor produce (224px agentview).
- HuggingFace auth: if `HF_HOME` is customized (e.g. on `/work`), ensure `$HF_HOME/token` exists before downloading (see user global rule), else downloads run unauthenticated and stall.
- Follow existing repo patterns; prefer small focused new files over editing large ones. New LIBERO modules live beside their nav siblings.

---

## Milestone 0 — Environment & assets

### Task 0.1: Provision the LIBERO environment and download the base checkpoint

**Files:**
- Create: `docs/superpowers/notes/phase1-env.md` (record exact versions/paths that worked)

**Interfaces:**
- Produces: a Python env in which `import experiments.robot.openvla_utils` works, LIBERO sim imports, and the base checkpoint is cached locally.

- [ ] **Step 1: Record the current env and GPU**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```
Expected: prints a torch version and a CUDA device name (on Nano4/Nano5 GPU). If CUDA is False, stop and provision a GPU node (see `~/Codes/console/NCHC_SERVER_README.md`).

- [ ] **Step 2: Install project + eval dependencies**

Run (in the project's venv):
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
pip install -e . 2>&1 | tail -5
pip install "transformers>=4.40" "tensorflow-datasets" "efficientnet_pytorch" "wandb" "imageio" "imageio-ffmpeg" 2>&1 | tail -5
```
Expected: installs complete without error. Record resolved versions into `docs/superpowers/notes/phase1-env.md`.

- [ ] **Step 3: Install LIBERO + robosuite**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO 2>/dev/null || echo "exists"
pip install -e ../LIBERO 2>&1 | tail -5
python -c "from libero.libero import benchmark; print('LIBERO ok', list(benchmark.get_benchmark_dict().keys())[:5])"
```
Expected: prints `LIBERO ok [...]` including a `libero_spatial` entry. If robosuite/mujoco errors appear, resolve them here (headless rendering may need `MUJOCO_GL=egl`); record the working env vars in the notes file.

- [ ] **Step 4: Confirm HF auth and download the base checkpoint**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
python -c "import os; h=os.environ.get('HF_HOME'); import pathlib; print('HF_HOME', h); print('token_present', pathlib.Path((h or os.path.expanduser('~/.cache/huggingface'))+'/token').exists())"
python -c "from huggingface_hub import snapshot_download; p=snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial'); print('CKPT', p)"
```
Expected: `token_present True` (if False, `cp ~/.cache/huggingface/token \"$HF_HOME/token\"` first), then `CKPT <path>` with the checkpoint materialized.

- [ ] **Step 5: Commit the env notes**

```bash
cd /home/chungyili/Codes/AsyncVLA-libero
git add docs/superpowers/notes/phase1-env.md
git commit -m "chore: record Phase-1 LIBERO env and asset provisioning"
```

---

## Milestone 1 — Retarget constants & model modules (pure, TDD)

### Task 1.1: Set LIBERO action constants for this branch

**Files:**
- Modify: `prismatic/vla/constants.py` (the `constants` dict, `ACTION_DIM`/`NUM_ACTIONS_CHUNK`)
- Test: `tests/test_constants_libero.py`

**Interfaces:**
- Produces: module-level `ACTION_DIM == 7`, `NUM_ACTIONS_CHUNK == 8` used by `get_current_action_mask`, `get_next_actions_mask`, datasets, and model heads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants_libero.py
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

def test_libero_action_constants():
    assert ACTION_DIM == 7, "LIBERO uses 7-DoF actions (6 EEF deltas + gripper)"
    assert NUM_ACTIONS_CHUNK == 8, "OpenVLA-OFT LIBERO uses an 8-step action chunk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_constants_libero.py -v`
Expected: FAIL — `assert 4 == 7` (nav default is `ACTION_DIM = 4`).

- [ ] **Step 3: Edit the constants dict**

In `prismatic/vla/constants.py`, set the values in the `constants` dict:
```python
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "POSE_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
```
(Only `ACTION_DIM` changes from 4→7; `POSE_DIM` is unused in the LIBERO path but keep it consistent.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_constants_libero.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prismatic/vla/constants.py tests/test_constants_libero.py
git commit -m "feat: retarget action constants to LIBERO (ACTION_DIM=7)"
```

### Task 1.2: Add a manipulation Edge Adapter (8×7 output)

**Files:**
- Modify: `prismatic/models/small_head.py` (add `Edge_adapter_manip`)
- Test: `tests/test_edge_adapter_manip.py`

**Interfaces:**
- Consumes: `NUM_ACTIONS_CHUNK`, `ACTION_DIM` from `prismatic.vla.constants`; `MultiLayerDecoder_trans` (already imported in `small_head.py`).
- Produces: `Edge_adapter_manip(obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=2, mha_ff_dim_factor=4, action_dim=ACTION_DIM)`; `forward(obs_img, past_img, vla_feature) -> Tensor[B, NUM_ACTIONS_CHUNK, action_dim]`. `obs_img`/`past_img` are `[B,3,96,96]`; `vla_feature` is `[B, NUM_ACTIONS_CHUNK, obs_encoding_size]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_adapter_manip.py
import torch
from prismatic.models.small_head import Edge_adapter_manip
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def test_edge_adapter_manip_output_shape():
    B, D = 2, 512
    edge = Edge_adapter_manip(obs_encoding_size=D)
    obs = torch.randn(B, 3, 96, 96)
    past = torch.randn(B, 3, 96, 96)
    vla_feature = torch.randn(B, NUM_ACTIONS_CHUNK, D)
    out = edge(obs, past, vla_feature)
    assert out.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM), out.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_edge_adapter_manip.py -v`
Expected: FAIL — `ImportError: cannot import name 'Edge_adapter_manip'`.

- [ ] **Step 3: Implement `Edge_adapter_manip`**

Append to `prismatic/models/small_head.py` (this is `Edge_adapter` with the final head width and reshape parametrized by `action_dim`):
```python
class Edge_adapter_manip(nn.Module):
    """AsyncVLA Edge Adapter retargeted for manipulation: predicts an
    (NUM_ACTIONS_CHUNK x action_dim) action chunk instead of 2D waypoints."""
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.action_dim = action_dim

        self.cat_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.num_cat_features = self.cat_encoder._fc.in_features
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        self.num_obs_features = self.obs_encoder._fc.in_features

        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size else nn.Identity()
        )
        self.compress_cat_enc = (
            nn.Linear(self.num_cat_features, self.obs_encoding_size)
            if self.num_cat_features != self.obs_encoding_size else nn.Identity()
        )

        self.decoder = MultiLayerDecoder_trans(
            embed_dim=self.obs_encoding_size,
            seq_len=NUM_ACTIONS_CHUNK + 1 + 1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(self.obs_encoding_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS_CHUNK * action_dim),
        )

    def forward(self, obs_img, past_img, vla_feature):
        batch_size = obs_img.shape[0]
        cat_img = torch.cat((obs_img, past_img), dim=1)
        cat_encoding = self.cat_encoder.extract_features(cat_img)
        cat_encoding = self.cat_encoder._avg_pooling(cat_encoding).flatten(start_dim=1)
        cat_encoding = self.compress_cat_enc(cat_encoding)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding).flatten(start_dim=1)
        obs_encoding = self.compress_obs_enc(obs_encoding)

        tokens = torch.cat(
            (vla_feature, obs_encoding.unsqueeze(1), cat_encoding.unsqueeze(1)), dim=1
        )
        tokens = self.decoder(tokens)[:, -2:-1, :]
        x = tokens.reshape(tokens.shape[0], -1)
        action_pred = self.action_predictor(x).reshape(batch_size, NUM_ACTIONS_CHUNK, self.action_dim)
        return action_pred
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_edge_adapter_manip.py -v`
Expected: PASS (output shape `(2, 8, 7)`).

- [ ] **Step 5: Commit**

```bash
git add prismatic/models/small_head.py tests/test_edge_adapter_manip.py
git commit -m "feat: add Edge_adapter_manip with 8x7 chunk output"
```

### Task 1.3: Configure the feature projector for LIBERO dims

**Files:**
- Test: `tests/test_proj_actiontokens_libero.py`

**Interfaces:**
- Consumes: existing `prismatic.models.small_head.Proj_Actiontokens(input_dim, hidden_dim, action_dim)`, `predict_action(actions_hidden_states[B, NUM_ACTIONS_CHUNK*ACTION_DIM, D], taskid[B]) -> [B, NUM_ACTIONS_CHUNK, action_dim]`.
- Produces: confirmed contract that with `action_dim=512`, the projector maps base hidden states to a `vla_feature` of `[B, NUM_ACTIONS_CHUNK, 512]` consumable by `Edge_adapter_manip`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proj_actiontokens_libero.py
import torch
from prismatic.models.small_head import Proj_Actiontokens
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def test_projector_emits_vla_feature():
    B, D, EDGE_DIM = 2, 4096, 512
    proj = Proj_Actiontokens(input_dim=D, hidden_dim=D, action_dim=EDGE_DIM)
    hidden = torch.randn(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
    taskid = torch.zeros(B)
    feat = proj.predict_action(hidden, taskid)
    assert feat.shape == (B, NUM_ACTIONS_CHUNK, EDGE_DIM), feat.shape
```

- [ ] **Step 2: Run test to verify it fails or errors**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_proj_actiontokens_libero.py -v`
Expected: If it errors on an internal dim, note the exact message; the fix is Step 3. If `Proj_Actiontokens` already accepts these dims, the test may PASS immediately — that is an acceptable green (skip Step 3).

- [ ] **Step 3: If needed, adapt `Proj_Actiontokens`**

If Step 2 failed on `MLPResNet_idcat` input width, confirm `input_dim*ACTION_DIM` matches `hidden.reshape(B, NUM_ACTIONS_CHUNK, -1)`'s last dim (`ACTION_DIM*D`). Adjust the `Proj_Actiontokens.__init__` `input_dim` wiring so `predict_action` reshapes `hidden` to `(B, NUM_ACTIONS_CHUNK, ACTION_DIM*D)` before `self.model`. Keep the change minimal and additive.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_proj_actiontokens_libero.py -v`
Expected: PASS (shape `(2, 8, 512)`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_proj_actiontokens_libero.py prismatic/models/small_head.py
git commit -m "test: pin Proj_Actiontokens -> vla_feature contract for LIBERO"
```

---

## Milestone 2 — Frozen base wrapper + hidden-state extraction

### Task 2.1: Base loader + config for LIBERO-Spatial

**Files:**
- Create: `experiments/robot/libero/__init__.py` (empty)
- Create: `experiments/robot/libero/base_config.py`
- Test: `tests/test_base_load.py` (GPU/network integration test; marked slow)

**Interfaces:**
- Consumes: `experiments.robot.openvla_utils.get_vla`, `get_processor`, `get_proprio_projector`.
- Produces: `LiberoBaseConfig` dataclass with fields `get_vla`/`get_processor` require — at minimum `pretrained_checkpoint: str`, `use_film: bool`, `num_images_in_input: int`, `load_in_8bit: bool=False`, `load_in_4bit: bool=False`, `use_proprio: bool=True`, `lora_rank: int=0`. `build_frozen_base(cfg) -> (vla, processor, proprio_projector)` with `vla.requires_grad_(False)`.

- [ ] **Step 1: Discover the exact cfg fields `get_vla`/FiLM expect**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
grep -n "cfg\." experiments/robot/openvla_utils.py | sed -n '1,60p'
```
Expected: a list of every `cfg.<field>` accessed (e.g. `pretrained_checkpoint`, `use_film`, `num_images_in_input`, `load_in_8bit`, `load_in_4bit`, `lora_rank`, `use_proprio`). Copy the full set into `base_config.py`.

- [ ] **Step 2: Write `base_config.py`**

```python
# experiments/robot/libero/base_config.py
from dataclasses import dataclass

@dataclass
class LiberoBaseConfig:
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    use_film: bool = True
    num_images_in_input: int = 2          # agentview + wrist, per OpenVLA-OFT LIBERO
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 0
    # NOTE: add any additional fields surfaced by Step 1 with their default values.

def build_frozen_base(cfg: LiberoBaseConfig):
    from experiments.robot.openvla_utils import get_vla, get_processor, get_proprio_projector
    vla = get_vla(cfg)
    vla.requires_grad_(False)
    vla.eval()
    processor = get_processor(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
        proprio_projector.requires_grad_(False)
    return vla, processor, proprio_projector
```
(Set `use_film`/`num_images_in_input`/`proprio_dim` to whatever the LIBERO-Spatial checkpoint expects — verify against Step-1 output and the checkpoint's `config.json`.)

- [ ] **Step 3: Write the integration test**

```python
# tests/test_base_load.py
import pytest, torch
pytestmark = pytest.mark.slow

def test_frozen_base_loads_and_has_no_grad():
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
    vla, processor, proprio = build_frozen_base(LiberoBaseConfig())
    assert all(not p.requires_grad for p in vla.parameters())
    assert hasattr(vla, "llm_dim") and vla.llm_dim > 0
```

- [ ] **Step 4: Run the test (needs GPU + downloaded checkpoint)**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_base_load.py -v -m slow`
Expected: PASS — model loads, all params frozen. If a `cfg.<field>` AttributeError appears, add that field (Step 1) and re-run.

- [ ] **Step 5: Commit**

```bash
git add experiments/robot/libero/__init__.py experiments/robot/libero/base_config.py tests/test_base_load.py
git commit -m "feat: frozen OpenVLA-OFT LIBERO base loader"
```

### Task 2.2: Extract action-token hidden states → `vla_feature`

**Files:**
- Create: `experiments/robot/libero/base_features.py`
- Test: `tests/test_base_features.py` (slow)

**Interfaces:**
- Consumes: a loaded `vla`, `processor`, `proprio_projector`; `prismatic.training.train_utils.get_current_action_mask`, `get_next_actions_mask`; `NUM_ACTIONS_CHUNK`, `ACTION_DIM`.
- Produces: `extract_actions_hidden_states(vla, batch, proprio_projector, num_patches) -> Tensor[B, NUM_ACTIONS_CHUNK*ACTION_DIM, D]`, mirroring `inference/run_asyncvla.py::run_forward_pass` lines 521–553. `num_patches` = number of vision patch tokens (read from the model / processor).

- [ ] **Step 1: Discover how the base is called for LIBERO (obs → batch → hidden states)**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
sed -n '515,556p' inference/run_asyncvla.py
grep -rn "num_patches\|NUM_PATCHES\|output_hidden_states\|predict_action" experiments/robot/openvla_utils.py
```
Expected: confirms the forward-call kwargs (`input_ids`, `attention_mask`, `pixel_values`, `proprio`, `proprio_projector`, `use_film`, `output_hidden_states=True`) and how `num_patches` is obtained. Record the LIBERO instruction-prompt format used by the stock model (needed to build `input_ids`).

- [ ] **Step 2: Implement `extract_actions_hidden_states`**

```python
# experiments/robot/libero/base_features.py
import torch
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

@torch.no_grad()
def extract_actions_hidden_states(vla, batch, proprio_projector, num_patches, device):
    output = vla(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
        labels=batch["labels"].to(device),
        output_hidden_states=True,
        proprio=batch.get("proprio", None),
        proprio_projector=proprio_projector,
        use_film=True,
    )
    last_hidden = output.hidden_states[-1]
    text_hidden = last_hidden[:, num_patches:-1]
    gt_token_ids = batch["labels"][:, 1:].to(device)
    cur_mask = get_current_action_mask(gt_token_ids)
    nxt_mask = get_next_actions_mask(gt_token_ids)
    bsz = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden[cur_mask | nxt_mask]
        .reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return actions_hidden_states
```
(Adjust the forward kwargs to match Step-1's exact signature for the OpenVLA-OFT LIBERO model.)

- [ ] **Step 3: Write the shape test**

```python
# tests/test_base_features.py
import pytest, torch
pytestmark = pytest.mark.slow

def test_actions_hidden_states_shape():
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
    from experiments.robot.libero.base_features import extract_actions_hidden_states
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
    vla, processor, proprio = build_frozen_base(LiberoBaseConfig())
    # Build a single-sample batch via the shared collator (Task 4.1 provides make_base_batch).
    from prismatic.vla.datasets.libero_dataset import make_dummy_base_batch
    batch, num_patches = make_dummy_base_batch(processor)
    hs = extract_actions_hidden_states(vla, batch, proprio, num_patches, device=next(vla.parameters()).device)
    assert hs.shape[0] == 1 and hs.shape[1] == NUM_ACTIONS_CHUNK * ACTION_DIM
```

- [ ] **Step 4: Run the test**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_base_features.py -v -m slow`
Expected: PASS — hidden states slice to `(1, 56, D)`. (Depends on `make_dummy_base_batch` from Task 4.1; if running Milestones out of order, implement that helper first.)

- [ ] **Step 5: Commit**

```bash
git add experiments/robot/libero/base_features.py tests/test_base_features.py
git commit -m "feat: extract action-token hidden states from frozen LIBERO base"
```

---

## Milestone 3 — Stock LIBERO eval harness (topline)

### Task 3.1: Port the stock OpenVLA-OFT LIBERO eval into this repo

**Files:**
- Create: `experiments/robot/libero/run_libero_eval.py`
- Create: `experiments/robot/libero/libero_utils.py` (env creation, obs→image helpers)

**Interfaces:**
- Consumes: `moojink/openvla-oft` repo's `experiments/robot/libero/run_libero_eval.py` + `libero_utils.py` as the reference implementation; the frozen base from Task 2.1.
- Produces: `python -m experiments.robot.libero.run_libero_eval --mode stock --task_suite libero_spatial --num_trials_per_task 1` running ≥1 episode and printing a per-task success flag.

- [ ] **Step 1: Fetch the reference eval code**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
git clone https://github.com/moojink/openvla-oft.git ../openvla-oft 2>/dev/null || echo "exists"
ls ../openvla-oft/experiments/robot/libero/
sed -n '1,80p' ../openvla-oft/experiments/robot/libero/run_libero_eval.py
```
Expected: lists `run_libero_eval.py`, `libero_utils.py`; prints the eval entrypoint. This is the porting reference (read-only; it is NOT `async_pi0`).

- [ ] **Step 2: Port `libero_utils.py` and a `stock`-mode `run_libero_eval.py`**

Copy `libero_utils.py` (env init, `get_libero_env`, `get_libero_image`, `save_rollout_video`, resize helpers) into `experiments/robot/libero/`. Port `run_libero_eval.py` trimmed to a single `--mode stock` path that: loads the frozen base (Task 2.1), for each of `--num_trials_per_task` rollouts per LIBERO-Spatial task, steps the env using the base's **native** `predict_action` (open-loop over the 8-step chunk exactly as the reference does), and records success. Keep argument names (`--task_suite`, `--num_trials_per_task`, `--seed`) identical to the reference. Adapt imports to this repo's `openvla_utils`.

- [ ] **Step 3: Smoke-run one episode**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval --mode stock --task_suite libero_spatial --num_trials_per_task 1 --num_tasks 1
```
Expected: one episode completes, prints `success=True/False` and an overall SR line. Fix any obs-key / normalization / prompt mismatches until it runs (these are the real integration points).

- [ ] **Step 4: Commit**

```bash
git add experiments/robot/libero/run_libero_eval.py experiments/robot/libero/libero_utils.py
git commit -m "feat: port stock OpenVLA-OFT LIBERO-Spatial eval harness"
```

### Task 3.2: Establish `SR_topline` (full protocol) — GATING

**Files:**
- Modify: `experiments/robot/libero/run_libero_eval.py` (add wandb logging of per-task + overall SR)

**Interfaces:**
- Produces: `SR_topline` for LIBERO-Spatial (10 tasks × 50 rollouts) logged to wandb; recorded in `docs/superpowers/notes/phase1-results.md`.

- [ ] **Step 1: Add wandb SR logging**

In `run_libero_eval.py`, after all rollouts, log `{"stock/sr_overall": sr, "stock/sr_<task>": sr_task, ...}` to a wandb run (`project` per rule). Write `docs/superpowers/notes/phase1-results.md` with the number.

- [ ] **Step 2: Run the full stock protocol**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval --mode stock --task_suite libero_spatial --num_trials_per_task 50
```
Expected: overall SR printed and logged. Sanity gate: `SR_topline` should be in the published OpenVLA-OFT LIBERO-Spatial range (~0.9+). If it is far below, STOP and debug base loading / prompt / normalization before proceeding — the edge cannot be validated against a broken topline.

- [ ] **Step 3: Commit results**

```bash
git add experiments/robot/libero/run_libero_eval.py docs/superpowers/notes/phase1-results.md
git commit -m "feat: log stock LIBERO-Spatial topline SR to wandb"
```

---

## Milestone 4 — LIBERO dataset + edge training

### Task 4.1: LIBERO-Spatial dataset (current+previous frame, GT chunk, base batch)

**Files:**
- Create: `prismatic/vla/datasets/libero_dataset.py`
- Test: `tests/test_libero_dataset.py`

**Interfaces:**
- Consumes: LIBERO-Spatial RLDS demos (`modified_libero_rlds`, downloadable per the openvla-oft README) OR the raw LIBERO hdf5 demos; `processor`, `ActionTokenizer`, prompt builder; `NUM_ACTIONS_CHUNK`, `ACTION_DIM`.
- Produces:
  - `LiberoSpatialDataset(...)` yielding a dict per item: `input_ids`, `attention_mask`, `pixel_values`, `labels`, `proprio` (base inputs), `obs_img_96 [3,96,96]`, `past_img_96 [3,96,96]`, `gt_action_chunk [NUM_ACTIONS_CHUNK, ACTION_DIM]`.
  - `make_dummy_base_batch(processor) -> (batch_dict, num_patches)` used by Task 2.2's test.
  - A collate function producing batched tensors.

- [ ] **Step 1: Resolve the data source**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
sed -n '/libero/,+3p' prismatic/vla/datasets/rlds/oxe/configs.py | head -40
grep -rn "libero" ../openvla-oft/README.md | head
```
Expected: identify the exact RLDS dataset name/keys for LIBERO-Spatial (image key, proprio key, action key) and the download instructions. Record the chosen source in `docs/superpowers/notes/phase1-data.md`.

- [ ] **Step 2: Write the dataset shape test (uses a tiny local sample)**

```python
# tests/test_libero_dataset.py
import torch
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def test_libero_sample_shapes(tmp_path):
    from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset
    ds = LiberoSpatialDataset(split="train", max_samples=4)  # small/dummy-safe
    item = ds[0]
    assert item["obs_img_96"].shape == (3, 96, 96)
    assert item["past_img_96"].shape == (3, 96, 96)
    assert item["gt_action_chunk"].shape == (NUM_ACTIONS_CHUNK, ACTION_DIM)
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in item
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_libero_dataset.py -v`
Expected: FAIL — module/class not present.

- [ ] **Step 4: Implement `LiberoSpatialDataset` + helpers**

Model it on `prismatic/vla/datasets/dummy_dataset.py` and `inference/run_asyncvla.py::data_transformer_asyncvla`: build the base `input_ids`/`labels` by tokenizing the LIBERO instruction + the 8×7 action tokens (via `ActionTokenizer`, using the checkpoint's `dataset_statistics.json` for normalization); produce `pixel_values` via the processor; produce `obs_img_96`/`past_img_96` by resizing consecutive agentview frames to 96×96 and normalizing with ImageNet mean/std (reuse the `transform` from `run_asyncvla.py`). For the first frame of an episode, `past_img_96 = obs_img_96`. Implement `make_dummy_base_batch(processor)` returning a single synthetic batch + `num_patches` for smoke tests.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_libero_dataset.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add prismatic/vla/datasets/libero_dataset.py tests/test_libero_dataset.py docs/superpowers/notes/phase1-data.md
git commit -m "feat: LIBERO-Spatial dataset with current/previous frame + GT chunk"
```

### Task 4.2: Edge training script (frozen base, L1 chunk loss)

**Files:**
- Create: `vla-scripts/train_asyncvla_libero.py`
- Test: `tests/test_train_overfit.py` (slow)

**Interfaces:**
- Consumes: `build_frozen_base`, `extract_actions_hidden_states`, `Proj_Actiontokens`, `Edge_adapter_manip`, `LiberoSpatialDataset`.
- Produces: a training entrypoint (`torchrun`-compatible) that freezes the base and optimizes only `{Edge_adapter_manip, Proj_Actiontokens}` with an L1 loss on the predicted 8×7 chunk vs GT; saves `shead--<step>_checkpoint.pt` and `proj--<step>_checkpoint.pt`; logs `train/l1_loss` to wandb.

- [ ] **Step 1: Write the overfit integration test**

```python
# tests/test_train_overfit.py
import pytest, torch
pytestmark = pytest.mark.slow

def test_edge_overfits_tiny_batch():
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    losses = train_one_batch_smoke(num_iters=50)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_train_overfit.py -v -m slow`
Expected: FAIL — `train_one_batch_smoke` not defined.

- [ ] **Step 3: Implement the training script + `train_one_batch_smoke`**

Adapt `vla-scripts/train_asyncvla.py` structure (config dataclass, DDP init, wandb, optimizer, save helpers) but: remove LoRA and all nav datasets/loss terms; load the frozen base via `build_frozen_base`; construct `Proj_Actiontokens(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=512)` and `Edge_adapter_manip(obs_encoding_size=512)`; the per-step forward = `extract_actions_hidden_states` → `proj.predict_action` → `edge(obs_96, past_96, vla_feature)`; `loss = F.l1_loss(pred_chunk, gt_chunk)`; `optimizer = AdamW(list(edge.parameters()) + list(proj.parameters()), lr=1e-4)`. Expose `train_one_batch_smoke(num_iters)` that runs the forward/backward on one fixed batch and returns the loss list.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chungyili/Codes/AsyncVLA-libero && python -m pytest tests/test_train_overfit.py -v -m slow`
Expected: PASS — final loss < half the initial loss (edge+projector can fit one batch).

- [ ] **Step 5: Commit**

```bash
git add vla-scripts/train_asyncvla_libero.py tests/test_train_overfit.py
git commit -m "feat: edge+projector training on LIBERO (frozen base, L1 chunk loss)"
```

### Task 4.3: Full training run on LIBERO-Spatial

**Files:**
- Modify: `docs/superpowers/notes/phase1-results.md` (record run id, steps, final loss, ckpt path)

**Interfaces:**
- Produces: trained `shead--<step>_checkpoint.pt` + `proj--<step>_checkpoint.pt` for `edge`-mode eval.

- [ ] **Step 1: Launch training with wandb**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_asyncvla_libero.py \
  --wandb_entity <ENTITY> --wandb_project asyncvla-libero --max_steps <N> --batch_size <B>
```
Expected: `train/l1_loss` decreases in wandb; checkpoints written under `runs/`. Choose `N`/`B` for the available GPU; document them.

- [ ] **Step 2: Record the run**

Append run id, final loss, and checkpoint paths to `docs/superpowers/notes/phase1-results.md`; commit.

```bash
git add docs/superpowers/notes/phase1-results.md
git commit -m "chore: record LIBERO edge training run"
```

---

## Milestone 5 — Edge eval + topline comparison

### Task 5.1: Add `edge` mode to the eval harness

**Files:**
- Modify: `experiments/robot/libero/run_libero_eval.py` (add `--mode edge`)
- Create: `experiments/robot/libero/edge_policy.py`

**Interfaces:**
- Consumes: frozen base + `extract_actions_hidden_states`; trained `Proj_Actiontokens` + `Edge_adapter_manip` checkpoints; the current + previous 96px agentview frame from the env.
- Produces: `EdgePolicy.act(obs) -> action_chunk[8,7]` mirroring `run_asyncvla.py::run_forward_pass` (base → projector → `vla_feature`; edge on current+previous frame → chunk). `--mode edge` runs the same rollout protocol as `stock`.

- [ ] **Step 1: Implement `EdgePolicy`**

Create `edge_policy.py`: load frozen base + `proj` + `edge` checkpoints; keep a one-frame history buffer for `past_img_96`; each control step build the base batch from the current obs, run `extract_actions_hidden_states` → `proj.predict_action` → `edge(obs_96, past_96, vla_feature)` → unnormalize with the checkpoint stats → return the 8×7 chunk. Execute the chunk open-loop exactly as `stock` mode does (same receding/open-loop cadence as the reference — synchronous; NOT the Phase-2 async loop).

- [ ] **Step 2: Wire `--mode edge` and smoke-run**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval --mode edge --task_suite libero_spatial --num_trials_per_task 1 --num_tasks 1 \
  --edge_ckpt runs/<run>/shead--<step>_checkpoint.pt --proj_ckpt runs/<run>/proj--<step>_checkpoint.pt
```
Expected: one episode completes end-to-end through the edge policy and prints `success=...`.

- [ ] **Step 3: Commit**

```bash
git add experiments/robot/libero/run_libero_eval.py experiments/robot/libero/edge_policy.py
git commit -m "feat: edge-mode LIBERO eval (base->projector->edge)"
```

### Task 5.2: Full edge protocol + report the SR gap

**Files:**
- Modify: `docs/superpowers/notes/phase1-results.md`

**Interfaces:**
- Produces: `SR_ours` (500 trials) and the reported gap `SR_topline − SR_ours`, logged to wandb alongside the Task 3.2 topline.

- [ ] **Step 1: Run the full edge protocol**

Run:
```bash
cd /home/chungyili/Codes/AsyncVLA-libero
MUJOCO_GL=egl python -m experiments.robot.libero.run_libero_eval --mode edge --task_suite libero_spatial --num_trials_per_task 50 \
  --edge_ckpt runs/<run>/shead--<step>_checkpoint.pt --proj_ckpt runs/<run>/proj--<step>_checkpoint.pt
```
Expected: overall + per-task `SR_ours` printed and logged to wandb as `edge/sr_overall`, `edge/sr_<task>`.

- [ ] **Step 2: Report the comparison**

Log a wandb summary panel with `stock/sr_overall`, `edge/sr_overall`, and `gap = stock − edge` (overall + per-task). Write the final table (topline vs ours vs gap) into `docs/superpowers/notes/phase1-results.md`. No hard pass/fail — the gap is the Phase-1 deliverable.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/phase1-results.md
git commit -m "docs: report LIBERO-Spatial SR_ours vs SR_topline gap (Phase 1 result)"
```

---

## Self-Review

**Spec coverage:**
- §2/§4 modules (base, projector, edge, constants) → Tasks 1.1, 1.2, 1.3, 2.1, 2.2. ✔
- §5 training (frozen base, consecutive frames, L1 chunk loss, wandb, DDP) → Tasks 4.1, 4.2, 4.3. ✔
- §5 evaluation (two modes, one protocol, 500 trials) + §3 topline + gap reporting → Tasks 3.1, 3.2, 5.1, 5.2. ✔
- §7 topline reproduction gate → Task 3.2 (explicit STOP gate). ✔
- §7 unit/integration/smoke tests → Tasks 1.1–1.3 (unit), 2.1/2.2/4.2 (integration), 3.1/5.1 (smoke). ✔
- §8 risks: checkpoint-format loading handled via existing `get_vla` (Task 2.1) + gate (3.2); hidden-state slicing verified (2.2). ✔
- §9 Phase-2 items (async loop, delay-k, latency) → intentionally absent. ✔

**Placeholder scan:** External-integration tasks (2.1 Step 1, 3.1, 4.1) use explicit discovery commands with expected output rather than fabricated APIs, because the exact LIBERO env obs-keys / OFT forward signature / RLDS keys must be read from the actual assets. These are actionable spikes with verification, not deferred work. Code we control (constants, edge, projector, tests) is written in full.

**Type consistency:** `vla_feature` is `[B, NUM_ACTIONS_CHUNK, 512]` everywhere (projector `action_dim=512` in 1.3, 4.2; consumed by `Edge_adapter_manip(obs_encoding_size=512)` in 1.2). `actions_hidden_states` is `[B, NUM_ACTIONS_CHUNK*ACTION_DIM, D]` in 2.2 and 1.3. `gt_action_chunk`/edge output are `[NUM_ACTIONS_CHUNK, ACTION_DIM]` = `[8,7]` in 1.2, 4.1, 4.2, 5.1. `build_frozen_base`, `extract_actions_hidden_states`, `make_dummy_base_batch`, `train_one_batch_smoke`, `EdgePolicy.act` names are used consistently across tasks.

**Known cross-milestone dependency:** Task 2.2's test imports `make_dummy_base_batch` from Task 4.1. If executing strictly in order, implement `make_dummy_base_batch` as part of Task 2.2 (a minimal synthetic batch) and let Task 4.1 supersede it, or reorder 4.1 before 2.2. Noted so the executor isn't surprised.
