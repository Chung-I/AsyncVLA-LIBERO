import pytest
import torch
import torch.nn.functional as F
from vla_scripts.train_asyncvla_libero import faithful_chunk_loss


def test_loss_is_their_exact_weighted_sum():
    """Weights are the original's: 0.5*15 (delta), 0.5 (traj). The smoothness term is
    DROPPED (see `faithful_chunk_loss` docstring for the closed-form derivation showing it
    is a mislabeled magnitude penalty, not a smoothness penalty).

    The two terms are recomputed here INDEPENDENTLY from `pred`/`gt` (not read back out of
    `faithful_chunk_loss`'s own `metrics` dict) -- so this test would catch
    `faithful_chunk_loss` swapping which tensors feed which term while keeping the key
    names the same (e.g. computing `mse_traj` on the raw actions instead of the cumsum).
    """
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    loss, m = faithful_chunk_loss(pred, gt)

    # Independent recomputation, mirroring `train_asyncvla.py:552,557` (delta / traj), NOT
    # `faithful_chunk_loss`'s internals.
    expected_delta = F.mse_loss(pred, gt)
    pred_traj = torch.cumsum(pred[..., :6], dim=1)
    gt_traj = torch.cumsum(gt[..., :6], dim=1)
    expected_traj = F.mse_loss(pred_traj, gt_traj)

    assert m["mse_delta"] == pytest.approx(expected_delta.item(), rel=1e-5)
    assert m["mse_traj"] == pytest.approx(expected_traj.item(), rel=1e-5)
    assert "mse_smooth" not in m

    expected_total = 0.5 * 15.0 * expected_delta + 0.5 * expected_traj
    assert loss.item() == pytest.approx(expected_total.item(), rel=1e-5)


def test_gripper_excluded_from_traj_term():
    """The gripper (dim 6) is an ABSOLUTE command -- it must not be integrated."""
    torch.manual_seed(0)
    pred, gt = torch.randn(2, 8, 7), torch.randn(2, 8, 7)
    _, m0 = faithful_chunk_loss(pred, gt)
    pred2 = pred.clone()
    pred2[..., 6] += 5.0                      # perturb ONLY the gripper dim
    _, m1 = faithful_chunk_loss(pred2, gt)
    assert m1["mse_delta"] != pytest.approx(m0["mse_delta"])      # delta term DOES see it
    assert m1["mse_traj"] == pytest.approx(m0["mse_traj"])        # traj term must NOT


def test_traj_vanishes_for_zero_eef_deltas():
    """Zero EEF deltas -> zero integrated trajectory (both pred and gt) -> traj term 0."""
    pred, gt = torch.zeros(2, 8, 7), torch.zeros(2, 8, 7)
    pred[..., 6] = 1.0                        # gripper nonzero, EEF all zero
    _, m = faithful_chunk_loss(pred, gt)
    assert m["mse_traj"] == pytest.approx(0.0)


def test_loss_is_differentiable():
    pred = torch.randn(2, 8, 7, requires_grad=True)
    loss, _ = faithful_chunk_loss(pred, torch.randn(2, 8, 7))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
