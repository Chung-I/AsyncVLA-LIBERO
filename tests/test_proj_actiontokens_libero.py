import torch
from prismatic.models.small_head import Proj_Actiontokens
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def test_projector_emits_vla_feature():
    B, D, EDGE_DIM = 2, 4096, 512
    proj = Proj_Actiontokens(input_dim=D, hidden_dim=D, action_dim=EDGE_DIM)
    hidden = torch.randn(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
    taskid = torch.zeros(B)
    feat = proj.predict_action(hidden, taskid)
    assert feat.shape == (B, NUM_ACTIONS_CHUNK, EDGE_DIM), feat.shape
