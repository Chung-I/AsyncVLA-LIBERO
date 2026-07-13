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
