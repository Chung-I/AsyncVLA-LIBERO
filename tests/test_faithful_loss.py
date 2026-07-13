import pytest
import torch
from vla_scripts.train_asyncvla_libero import faithful_chunk_loss


def test_loss_is_their_exact_weighted_sum():
    """Weights are the original's: 0.5*15 (delta), 0.5 (traj), 0.1 (smoothness)."""
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    loss, m = faithful_chunk_loss(pred, gt)
    expected = 0.5 * 15.0 * m["mse_delta"] + 0.5 * m["mse_traj"] + 0.1 * m["mse_smooth"]
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_gripper_excluded_from_traj_and_smoothness_terms():
    """The gripper (dim 6) is an ABSOLUTE command -- it must not be integrated."""
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    _, m0 = faithful_chunk_loss(pred, gt)
    pred2 = pred.clone()
    pred2[..., 6] += 5.0                      # perturb ONLY the gripper dim
    _, m1 = faithful_chunk_loss(pred2, gt)
    assert m1["mse_delta"] != pytest.approx(m0["mse_delta"])      # delta term DOES see it
    assert m1["mse_traj"] == pytest.approx(m0["mse_traj"])        # traj term must NOT
    assert m1["mse_smooth"] == pytest.approx(m0["mse_smooth"])    # smoothness must NOT


def test_smoothness_and_traj_vanish_for_zero_eef_deltas():
    """Zero EEF deltas -> zero integrated trajectory -> sm_ref == traj -> both terms 0."""
    pred, gt = torch.zeros(2, 8, 7), torch.zeros(2, 8, 7)
    pred[..., 6] = 1.0                        # gripper nonzero, EEF all zero
    _, m = faithful_chunk_loss(pred, gt)
    assert m["mse_traj"] == pytest.approx(0.0)
    assert m["mse_smooth"] == pytest.approx(0.0)


def test_loss_is_differentiable():
    pred = torch.randn(2, 8, 7, requires_grad=True)
    loss, _ = faithful_chunk_loss(pred, torch.randn(2, 8, 7))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
