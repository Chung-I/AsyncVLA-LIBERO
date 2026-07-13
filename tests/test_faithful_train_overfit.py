import pytest
pytestmark = pytest.mark.slow


def test_faithful_edge_overfits_tiny_batch():
    """Faithful config end-to-end: original Edge_adapter @1024/4/4, k_max=3, 3-term MSE,
    random-crop augmentation ON (`image_aug=True` -- the trainer's own default; leaving it
    False here would make this a functional duplicate of `test_train_overfit_delay`)."""
    from vla_scripts.train_asyncvla_libero import train_one_batch_smoke
    losses = train_one_batch_smoke(num_iters=200, delay_aware=True, k_max=3, image_aug=True)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
