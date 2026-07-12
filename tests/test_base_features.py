import pytest
import torch

pytestmark = pytest.mark.slow


def test_actions_hidden_states_shape():
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
    from experiments.robot.libero.base_features import extract_actions_hidden_states
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
    from prismatic.vla.datasets.libero_dataset import make_dummy_base_batch

    vla, processor, proprio = build_frozen_base(LiberoBaseConfig())
    batch, num_patches = make_dummy_base_batch(processor)

    assert num_patches == 513

    hs = extract_actions_hidden_states(
        vla, batch, proprio, num_patches, device=next(vla.parameters()).device
    )

    assert hs.shape == (1, NUM_ACTIONS_CHUNK * ACTION_DIM, vla.llm_dim)
    assert hs.shape[0] == 1 and hs.shape[1] == NUM_ACTIONS_CHUNK * ACTION_DIM


def test_actions_hidden_states_feed_proj_actiontokens():
    """Sanity cross-check: the extracted actions_hidden_states is a valid vla_feature
    for the edge-adapter's Proj_Actiontokens head (shape-compatible end to end)."""
    from experiments.robot.libero.base_config import LiberoBaseConfig, build_frozen_base
    from experiments.robot.libero.base_features import extract_actions_hidden_states
    from prismatic.models.small_head import Proj_Actiontokens
    from prismatic.vla.datasets.libero_dataset import make_dummy_base_batch

    vla, processor, proprio = build_frozen_base(LiberoBaseConfig())
    batch, num_patches = make_dummy_base_batch(processor)

    hs = extract_actions_hidden_states(
        vla, batch, proprio, num_patches, device=next(vla.parameters()).device
    )

    proj = Proj_Actiontokens(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=512)
    proj = proj.to(device=hs.device, dtype=torch.float32)
    taskid = torch.zeros(hs.shape[0], device=hs.device)

    with torch.no_grad():
        vla_feature = proj.predict_action(hs.float(), taskid)

    assert vla_feature.shape == (1, 8, 512)
