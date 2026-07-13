import pytest
import torch
import torch.nn.functional as F
from vla_scripts.train_asyncvla_libero import faithful_chunk_loss


def test_loss_is_their_exact_weighted_sum():
    """Weights are the original's: 0.5*15 (delta), 0.5 (traj), 0.1 (smoothness).

    The three terms are recomputed here INDEPENDENTLY from `pred`/`gt` (not read back out
    of `faithful_chunk_loss`'s own `metrics` dict) -- so this test would catch
    `faithful_chunk_loss` swapping which tensors feed which term while keeping the key
    names the same (e.g. computing `mse_traj` on the raw actions instead of the cumsum).
    """
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    loss, m = faithful_chunk_loss(pred, gt)

    # Independent recomputation, mirroring `train_asyncvla.py:552,557,559` (delta / traj /
    # smoothness), NOT `faithful_chunk_loss`'s internals.
    expected_delta = F.mse_loss(pred, gt)
    pred_traj = torch.cumsum(pred[..., :6], dim=1)
    gt_traj = torch.cumsum(gt[..., :6], dim=1)
    expected_traj = F.mse_loss(pred_traj, gt_traj)
    sm_ref = torch.cat([torch.zeros_like(pred_traj[:, :1]), pred_traj[:, :-1]], dim=1)
    expected_smooth = F.mse_loss(pred_traj, sm_ref)

    assert m["mse_delta"] == pytest.approx(expected_delta.item(), rel=1e-5)
    assert m["mse_traj"] == pytest.approx(expected_traj.item(), rel=1e-5)
    assert m["mse_smooth"] == pytest.approx(expected_smooth.item(), rel=1e-5)

    expected_total = 0.5 * 15.0 * expected_delta + 0.5 * expected_traj + 0.1 * expected_smooth
    assert loss.item() == pytest.approx(expected_total.item(), rel=1e-5)


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
