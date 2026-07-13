import torch
from prismatic.models.small_head import Edge_adapter
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj


def test_original_edge_adapter_outputs_libero_chunk():
    """The ORIGINAL Edge_adapter, at the ORIGINAL capacity, emits an 8x7 LIBERO chunk."""
    B, D = 2, 1024
    edge = Edge_adapter(
        obs_encoding_size=D, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4
    )
    out = edge(torch.randn(B, 3, 96, 96), torch.randn(B, 3, 96, 96), torch.randn(B, NUM_ACTIONS_CHUNK, D))
    assert out.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM) == (B, 8, 7), out.shape


def test_output_head_width_is_chunk_times_action_dim():
    """The one-line change: Linear(64, 8*4) -> Linear(64, NUM_ACTIONS_CHUNK*ACTION_DIM).
    Under the ORIGINAL nav constants (chunk=8, ACTION_DIM=4) this expression evaluates to
    32 -- byte-identical to the original literal `8 * 4` -- so the change is a semantic
    no-op for navigation. Under LIBERO constants (chunk=8, ACTION_DIM=7) it is 56."""
    edge = Edge_adapter(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4)
    assert edge.action_predictor[-1].out_features == NUM_ACTIONS_CHUNK * ACTION_DIM == 56


def test_edge_arch_from_config_nav_is_the_original_capacity():
    """Capacity comes from the ORIGINAL config file, not from our class defaults."""
    arch = EdgeArch.from_config_nav()
    assert (arch.obs_encoding_size, arch.mha_num_attention_heads,
            arch.mha_num_attention_layers, arch.mha_ff_dim_factor) == (1024, 4, 4, 4)


def test_build_edge_and_proj_at_faithful_capacity():
    edge, proj = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=EdgeArch.from_config_nav())
    assert isinstance(edge, Edge_adapter)
    assert edge.obs_encoding_size == 1024
    # projector token width MUST equal the edge token width (they are concatenated)
    hidden = torch.randn(2, NUM_ACTIONS_CHUNK * ACTION_DIM, 4096)
    feat = proj.predict_action(hidden, torch.zeros(2))
    assert feat.shape == (2, NUM_ACTIONS_CHUNK, 1024)
