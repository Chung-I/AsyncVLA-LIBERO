import torch
from prismatic.models.small_head import Edge_adapter_manip
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def test_edge_adapter_manip_output_shape():
    B, D = 2, 512
    edge = Edge_adapter_manip(obs_encoding_size=D)
    obs = torch.randn(B, 3, 96, 96)
    past = torch.randn(B, 3, 96, 96)
    vla_feature = torch.randn(B, NUM_ACTIONS_CHUNK, D)
    out = edge(obs, past, vla_feature)
    assert out.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM), out.shape
