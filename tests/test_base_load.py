import pytest

pytestmark = pytest.mark.slow


def test_frozen_base_loads_and_has_no_grad():
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base

    vla, processor, proprio = build_frozen_base(LiberoBaseConfig())

    assert all(not p.requires_grad for p in vla.parameters())
    assert hasattr(vla, "llm_dim") and vla.llm_dim > 0

    # use_proprio=True by default for this checkpoint -> proprio projector must load and be frozen.
    assert proprio is not None
    assert all(not p.requires_grad for p in proprio.parameters())

    assert processor is not None
