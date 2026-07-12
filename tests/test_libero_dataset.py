import pytest
import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def test_libero_sample_shapes(tmp_path):
    from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset

    ds = LiberoSpatialDataset(split="train", max_samples=4)  # small/dummy-safe
    assert len(ds) > 0
    item = ds[0]
    assert item["obs_img_96"].shape == (3, 96, 96)
    assert item["past_img_96"].shape == (3, 96, 96)
    assert item["gt_action_chunk"].shape == (NUM_ACTIONS_CHUNK, ACTION_DIM)
    for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
        assert k in item


def test_libero_first_frame_past_equals_obs():
    """For the first step of an episode, past_img_96 must equal obs_img_96 (no t-1 frame)."""
    from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset

    ds = LiberoSpatialDataset(split="train", max_samples=1)
    item = ds[0]
    assert torch.equal(item["obs_img_96"], item["past_img_96"])


def test_libero_pixel_values_has_two_images():
    """`pixel_values` must channel-stack agentview + wrist (2 images) for the frozen base."""
    from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset

    ds = LiberoSpatialDataset(split="train", max_samples=1)
    item = ds[0]
    # Fused vision backbone: each image occupies `pixel_values.shape[0] // 2` channels.
    assert item["pixel_values"].shape[0] % 2 == 0


def test_collate_libero_batch_stacks_fields():
    from prismatic.vla.datasets.libero_dataset import LiberoSpatialDataset, collate_libero_batch

    ds = LiberoSpatialDataset(split="train", max_samples=3)
    items = [ds[i] for i in range(len(ds))]
    batch = collate_libero_batch(items, pad_token_id=ds.processor.tokenizer.pad_token_id)

    assert batch["input_ids"].shape[0] == len(items)
    assert batch["gt_action_chunk"].shape == (len(items), NUM_ACTIONS_CHUNK, ACTION_DIM)
    assert batch["obs_img_96"].shape == (len(items), 3, 96, 96)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape


@pytest.mark.slow
def test_make_dummy_base_batch_action_mask_selects_56_through_real_base():
    """`make_dummy_base_batch` must produce a batch whose action-token layout, once run
    through the REAL frozen base's forward pass, makes the base's action masks select
    exactly NUM_ACTIONS_CHUNK * ACTION_DIM == 56 positions (the contract the next task's
    hidden-state extraction depends on).
    """
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.datasets.libero_dataset import make_dummy_base_batch

    vla, processor, proprio_projector = build_frozen_base(LiberoBaseConfig())
    device = next(vla.parameters()).device

    batch, num_patches = make_dummy_base_batch(processor)
    assert num_patches == 513

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["proprio"].to(torch.bfloat16).to(device),
            proprio_projector=proprio_projector,
            use_film=False,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    total_selected = (current_action_mask | next_actions_mask).sum().item()
    assert total_selected == NUM_ACTIONS_CHUNK * ACTION_DIM  # == 56

    last_hidden_states = out.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    assert text_hidden_states.shape[1] == ground_truth_token_ids.shape[1]

    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
        batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
    )
    assert actions_hidden_states.shape == (batch_size, 56, vla.llm_dim)
