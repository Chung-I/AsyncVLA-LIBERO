import pytest

pytestmark = pytest.mark.slow


def test_delay_aware_edge_overfits_tiny_batch():
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke

    # Random k in {0..15} across 200 iters must still drive loss well below half.
    losses = train_one_batch_smoke(num_iters=200, delay_aware=True, k_max=15)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
