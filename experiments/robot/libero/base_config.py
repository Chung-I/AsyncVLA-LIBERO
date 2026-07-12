r"""Frozen OpenVLA-OFT LIBERO-Spatial base loader.

`LiberoBaseConfig` supplies every `cfg.<field>` accessed by
`experiments.robot.openvla_utils.get_vla`, `get_processor`, and
`get_proprio_projector` (the three entry points `build_frozen_base` calls).
Discovered via:

    grep -n "cfg\." experiments/robot/openvla_utils.py

which surfaced, for those three functions:
    - cfg.pretrained_checkpoint  (get_vla, get_processor, get_proprio_projector)
    - cfg.load_in_8bit           (get_vla)
    - cfg.load_in_4bit           (get_vla)
    - cfg.use_film               (get_vla -> _apply_film_to_vla)
    - cfg.num_images_in_input    (get_vla -> vla.vision_backbone.set_num_images_in_input)
    - cfg.lora_rank              (get_vla -> _apply_film_to_vla, only reached if use_film=True)

Field values were then verified against the actual checkpoint
(`moojink/openvla-7b-oft-finetuned-libero-spatial`), not guessed:
    - Its HF Hub file listing has NO `*vision_backbone*checkpoint*` file, and its
      README.md "Quick Start" snippet instantiates `GenerateConfig` with
      `use_film=False`. Since `_apply_film_to_vla` would otherwise try (and fail,
      since `find_checkpoint_file` requires a local directory, not a Hub repo id)
      to load a FiLM-wrapped vision-backbone checkpoint that doesn't exist for
      this repo, `use_film=False` is required, not merely a stylistic default.
    - The same README snippet also sets `num_images_in_input=2` (agentview +
      wrist) and `use_proprio=True`.
    - The checkpoint's `proprio_projector--150000_checkpoint.pt` has
      `fc1.weight` of shape `[4096, 8]`, confirming `proprio_dim=8` and
      `llm_dim=4096` for this checkpoint.
    - `lora_rank` is unused for this checkpoint (only consulted inside
      `_apply_film_to_vla`, which is skipped when `use_film=False`); kept at 0
      as an inert default.
"""

from dataclasses import dataclass


@dataclass
class LiberoBaseConfig:
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    use_film: bool = False  # this checkpoint ships no FiLM-wrapped vision-backbone weights
    num_images_in_input: int = 2  # agentview + wrist, per the checkpoint's README Quick Start
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 0  # unused when use_film=False; kept for get_vla's cfg surface


# Proprio dim for this checkpoint's proprio_projector (verified against the
# checkpoint's proprio_projector--150000_checkpoint.pt: fc1.weight is [4096, 8]).
PROPRIO_DIM = 8


def build_frozen_base(cfg: LiberoBaseConfig):
    """Instantiate the frozen OpenVLA-OFT LIBERO base model + processor + proprio projector.

    Returns:
        (vla, processor, proprio_projector): `vla` and `proprio_projector` (if
        `cfg.use_proprio`) are fully frozen (`requires_grad_(False)`, `.eval()`).
        `proprio_projector` is `None` when `cfg.use_proprio` is False.
    """
    from experiments.robot.openvla_utils import get_proprio_projector, get_processor, get_vla

    vla = get_vla(cfg)
    vla.requires_grad_(False)
    vla.eval()

    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=PROPRIO_DIM)
        proprio_projector.requires_grad_(False)
        proprio_projector.eval()

    return vla, processor, proprio_projector
