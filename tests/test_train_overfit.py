import pytest, torch
pytestmark = pytest.mark.slow

def test_edge_overfits_tiny_batch():
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    # 200 iters clears the "< 0.5x initial loss" bar robustly for any seed (measured
    # ratio ~0.10-0.13 at 150 iters, ~0.03-0.05 at 300); 50 iters was flaky (~0.51-0.55).
    losses = train_one_batch_smoke(num_iters=200)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
