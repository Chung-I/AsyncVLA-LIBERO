"""CPU-only unit test for the `MultiStepLR` schedule added to `train_asyncvla_libero.py`
(C3): a 10x decay at `cfg.num_steps_before_decay`, mirroring the original
`vla-scripts/train_asyncvla.py:1059-1063` (`MultiStepLR(optimizer,
milestones=[cfg.num_steps_before_decay], gamma=0.1)`), which neither LIBERO branch had.

No model/dataset required: a tiny `nn.Linear` + AdamW is enough to exercise the scheduler.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR


def test_lr_decays_10x_at_milestone():
    model = nn.Linear(4, 4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    for step in range(10):
        if step < 5:
            assert optimizer.param_groups[0]["lr"] == 1e-4, step
        else:
            assert optimizer.param_groups[0]["lr"] == 1e-5, step
        optimizer.step()
        scheduler.step()


def test_config_has_lr_decay_fields_matching_50k_half_ratio():
    """`num_steps_before_decay` defaults to 25k -- half of our `max_steps=50_000`,
    preserving the original's ratio (`num_steps_before_decay=100_000` of
    `max_steps=200_000`) rather than its literal 100k value."""
    from vla_scripts.train_asyncvla_libero import AsyncVLALiberoConfig

    cfg = AsyncVLALiberoConfig()
    assert cfg.num_steps_before_decay == 25_000
    assert cfg.num_steps_before_decay == cfg.max_steps // 2
    assert cfg.lr_decay_gamma == 0.1
