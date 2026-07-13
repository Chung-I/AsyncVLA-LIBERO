import pytest
pytestmark = pytest.mark.slow


def test_faithful_edge_overfits_tiny_batch():
    """Faithful config end-to-end: original Edge_adapter @1024/4/4, k_max=3, 3-term MSE."""
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    losses = train_one_batch_smoke(num_iters=200, delay_aware=True, k_max=3)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
