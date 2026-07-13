import numpy as np
import pytest
import torch
from PIL import Image
from prismatic.vla.datasets.libero_dataset import (
    DEFAULT_DATA_DIR, H_RANDOM, V_RANDOM, LiberoSpatialDataset,
    _apply_crop, _has_tfrecords, _sample_crop_offsets, _to_base_image, _to_edge_frame,
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


def test_aug_is_deterministic_under_seed():
    """Two same-seeded dataset instances produce identical `_last_crop_offsets` and identical
    output tensors. This alone does NOT prove the box is shared across the four raw frames of
    a single sample -- see `test_aug_crops_all_four_frames_with_one_shared_box` for that.
    """
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    a = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    b = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                             image_aug=True, rng_seed=0)
    ia, ib = a[0], b[0]
    assert a._last_crop_offsets == b._last_crop_offsets
    assert torch.allclose(ia["obs_img_96"], ib["obs_img_96"])       # deterministic under seed
    assert torch.allclose(ia["pixel_values"], ib["pixel_values"])


def test_aug_crops_all_four_frames_with_one_shared_box():
    """Reconstruct the four raw frames (base agentview, base wrist, edge current, edge past)
    independently from the underlying episode and prove every one of them was cropped with
    the exact SAME box recorded in `_last_crop_offsets`. A per-frame (independently sampled)
    box, or a frame the augmentation forgot to crop, fails this test.
    """
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    ds = LiberoSpatialDataset(split="train", max_samples=8, image_aug=True, rng_seed=0)

    # idx=1 -> (ep_idx=0, t=1): the first sample with a genuine t-1 "past" frame. idx=0 has
    # t=0, where past_img_96 == obs_img_96 (see test_libero_first_frame_past_equals_obs), so
    # it can't distinguish a per-frame box on raw_prev from the shared one on raw_now.
    idx = 1
    ep_idx, t = ds._index[idx]
    assert t > 0

    item = ds[idx]
    voffset, hoffset = ds._last_crop_offsets
    assert (voffset, hoffset) != (0, 0)  # nonzero box, so the negative control below can bite

    ep = ds._episodes[ep_idx]
    raw_now = np.asarray(ep["image"][t])
    raw_wrist = np.asarray(ep["wrist_image"][t])   # delay_aware=False here, so t_base == t
    raw_prev = np.asarray(ep["image"][t - 1])

    # Negative control: the crop must actually remove pixels (rules out a no-op _apply_crop).
    cropped_now = _apply_crop(raw_now, voffset, hoffset)
    assert cropped_now.shape[0] < raw_now.shape[0] or cropped_now.shape[1] < raw_now.shape[1]

    cropped_wrist = _apply_crop(raw_wrist, voffset, hoffset)
    cropped_prev = _apply_crop(raw_prev, voffset, hoffset)

    # Base agentview + wrist: crop -> `_to_base_image` -> the processor's own image transform
    # -> compare against the `pixel_values` channel-stack (0:6 = primary, 6:12 = wrist, for the
    # fused dinosiglip 6-channel-per-image encoding; see test_libero_pixel_values_has_two_images).
    expected_primary_pv = ds.processor.image_processor.apply_transform(
        _to_base_image(cropped_now).convert("RGB")
    )
    expected_wrist_pv = ds.processor.image_processor.apply_transform(
        _to_base_image(cropped_wrist).convert("RGB")
    )
    assert torch.allclose(item["pixel_values"][:6], expected_primary_pv)
    assert torch.allclose(item["pixel_values"][6:], expected_wrist_pv)

    # Edge current + past.
    expected_obs = _to_edge_frame(Image.fromarray(cropped_now))
    expected_past = _to_edge_frame(Image.fromarray(cropped_prev))
    assert torch.allclose(item["obs_img_96"], expected_obs)
    assert torch.allclose(item["past_img_96"], expected_past)


def test_aug_off_by_default_and_changes_the_images_when_on():
    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")
    plain = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3)
    assert plain.image_aug is False                                  # eval/default path: NO crop
    augd = LiberoSpatialDataset(split="train", max_samples=8, delay_aware=True, k_max=3,
                                image_aug=True, rng_seed=1)
    # A nonzero crop must actually change the produced frames.
    assert not torch.allclose(plain[0]["obs_img_96"], augd[0]["obs_img_96"])
