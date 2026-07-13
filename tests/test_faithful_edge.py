import torch
from prismatic.models.small_head import Edge_adapter
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj


def test_original_edge_adapter_outputs_libero_chunk():
    """The ORIGINAL Edge_adapter, at the ORIGINAL capacity, with `action_dim=ACTION_DIM`
    passed explicitly (as `build_edge_and_proj` does for LIBERO), emits an 8x7 chunk."""
    B, D = 2, 1024
    edge = Edge_adapter(
        obs_encoding_size=D, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4,
        action_dim=ACTION_DIM,
    )
    out = edge(torch.randn(B, 3, 96, 96), torch.randn(B, 3, 96, 96), torch.randn(B, NUM_ACTIONS_CHUNK, D))
    assert out.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM) == (B, 8, 7), out.shape


def test_output_head_width_is_parameterized_not_hardcoded():
    """`Edge_adapter.action_predictor`'s output width is `NUM_ACTIONS_CHUNK * action_dim`,
    where `action_dim` is now a constructor kwarg. NOTE: upstream AsyncVLA (`main`) defines
    `ACTION_DIM = 4` -- the nav action width -- but the `asyncvla-libero` branch this branch
    forked from retargeted `prismatic/vla/constants.py`'s `ACTION_DIM`/`POSE_DIM` from 4 to 7
    for LIBERO's 7-DoF action space (commit `7954582`), so on THIS branch `ACTION_DIM` is 7
    and no longer means "the nav width". The head width must therefore NOT be derived from
    `ACTION_DIM` -- the default `action_dim=4` (32-wide) hardcodes the original nav width
    directly, independent of the retargeted constant, preserving the original literal
    `Linear(64, 8 * 4)` byte-for-byte for every original caller (`vla-scripts/train_asyncvla.py`,
    `inference/run_asyncvla.py`, which never pass `action_dim`); the LIBERO path passes
    `action_dim=ACTION_DIM` (=7, 56-wide) explicitly via `build_edge_and_proj`."""
    edge_original_default = Edge_adapter(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4)
    assert edge_original_default.action_predictor[-1].out_features == 8 * 4 == 32

    edge_libero = Edge_adapter(
        obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, action_dim=ACTION_DIM
    )
    assert edge_libero.action_predictor[-1].out_features == NUM_ACTIONS_CHUNK * ACTION_DIM == 56


def test_edge_arch_from_config_nav_is_the_original_capacity():
    """Capacity comes from the ORIGINAL config file, not from our class defaults."""
    arch = EdgeArch.from_config_nav()
    assert (arch.obs_encoding_size, arch.mha_num_attention_heads,
            arch.mha_num_attention_layers, arch.mha_ff_dim_factor) == (1024, 4, 4, 4)


def test_build_edge_and_proj_at_faithful_capacity():
    edge, proj = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=EdgeArch.from_config_nav())
    assert isinstance(edge, Edge_adapter)
    assert edge.obs_encoding_size == 1024
    # pins the LIBERO edge head width: build_edge_and_proj must pass action_dim=ACTION_DIM,
    # not fall back to Edge_adapter's own nav default (4) -- see F1 / test_faithful_edge.py's
    # test_output_head_width_is_parameterized_not_hardcoded for the full rationale.
    assert edge.action_predictor[-1].out_features == NUM_ACTIONS_CHUNK * ACTION_DIM  # 8 x 7 = 56
    # projector token width MUST equal the edge token width (they are concatenated)
    hidden = torch.randn(2, NUM_ACTIONS_CHUNK * ACTION_DIM, 4096)
    feat = proj.predict_action(hidden, torch.zeros(2))
    assert feat.shape == (2, NUM_ACTIONS_CHUNK, 1024)
