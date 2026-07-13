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


class _FixedDelayRNG:
    """Stub for `ds._delay_rng` whose `.randint(low, high)` always returns a fixed value,
    so `_sample_delay(t, k_max, rng) = min(fixed, t)` is deterministic for any `t >= fixed`.
    """

    def __init__(self, fixed: int):
        self._fixed = fixed

    def randint(self, low, high):
        return self._fixed


def test_delay_aware_routes_base_to_stale_frame_when_t_and_k_positive():
    # `ds[0]` always has t=0 (episodes start their (ep_idx, t) index at t=0), so k clamps to
    # 0 and the t_base != t routing path is never exercised there. Force a sample with t>0
    # and a deterministic k>0, then assert the base ("past") frame really comes from t-k
    # (not t, not t-1), while the edge's "current" frame stays at t and the two differ.
    from prismatic.vla.datasets.libero_dataset import DEFAULT_DATA_DIR, _has_tfrecords, _to_edge_frame
    import pytest
    from PIL import Image

    if not _has_tfrecords(DEFAULT_DATA_DIR):
        pytest.skip("real local RLDS shard not present")

    ds = LiberoSpatialDataset(split="train", max_samples=64, delay_aware=True, k_max=15)

    k = 5
    idx, ep_idx, t = None, None, None
    for i, (candidate_ep_idx, candidate_t) in enumerate(ds._index):
        if candidate_t < k:
            continue
        ep = ds._episodes[candidate_ep_idx]
        frame_t = np.asarray(ep["image"][candidate_t])
        frame_tmk = np.asarray(ep["image"][candidate_t - k])
        if not np.array_equal(frame_t, frame_tmk):
            idx, ep_idx, t = i, candidate_ep_idx, candidate_t
            break
    assert idx is not None, "no sample with t >= k and distinct frames t steps apart was found"
    assert t >= 5

    ep = ds._episodes[ep_idx]
    ds._delay_rng = _FixedDelayRNG(k)  # forces _sample_delay(t, 15, rng) == min(k, t) == k

    item = ds[idx]

    expected_past = _to_edge_frame(Image.fromarray(np.asarray(ep["image"][t - k])))
    expected_obs = _to_edge_frame(Image.fromarray(np.asarray(ep["image"][t])))

    # The base's "past" frame is routed from t-k, bit-for-bit -- not from t or t-1.
    assert torch.allclose(item["past_img_96"], expected_past)
    # The edge's "current" frame stays at t, unaffected by the delay.
    assert torch.allclose(item["obs_img_96"], expected_obs)
    # Since the raw frames at t and t-k differ, the routed tensors must differ too --
    # proving the delay actually shifted the base's frame back in time.
    assert not torch.allclose(item["obs_img_96"], item["past_img_96"])
