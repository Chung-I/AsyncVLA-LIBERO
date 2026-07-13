import numpy as np
import pytest
import torch
from prismatic.vla.datasets.libero_dataset import (
    DEFAULT_DATA_DIR, H_RANDOM, V_RANDOM, LiberoSpatialDataset,
    _apply_crop, _has_tfrecords, _sample_crop_offsets,
)


def test_crop_offsets_match_the_original_fractions():
    """Original: voffset = int(224 * v_random * rand()), v_random=0.2, h_random=0.1."""
    assert (V_RANDOM, H_RANDOM) == (0.2, 0.1)
    rng = np.random.RandomState(0)
    for _ in range(200):
        v, h = _sample_crop_offsets(256, 256, rng)
        assert 0 <= v < 0.2 * 256
        assert 0 <= h < 0.1 * 256


def test_apply_crop_is_symmetric():
    raw = np.zeros((256, 256, 3), dtype=np.uint8)
    out = _apply_crop(raw, voffset=10, hoffset=5)
    assert out.shape == (256 - 2 * 10, 256 - 2 * 5, 3)


def test_aug_uses_ONE_shared_box_and_is_deterministic_under_seed():
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    a = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    b = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    ia, ib = a[0], b[0]
    # One box per sample, recorded for inspection -- the SAME box drove all four images.
    assert a._last_crop_offsets == b._last_crop_offsets
    assert torch.allclose(ia["obs_img_96"], ib["obs_img_96"])       # deterministic under seed
    assert torch.allclose(ia["pixel_values"], ib["pixel_values"])


def test_aug_off_by_default_and_changes_the_images_when_on():
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    plain = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3)
    assert plain.image_aug is False                                  # eval/default path: NO crop
    augd = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                                image_aug=True, rng_seed=1)
    # A nonzero crop must actually change the produced frames.
    assert not torch.allclose(plain[0]["obs_img_96"], augd[0]["obs_img_96"])
