from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

def test_libero_action_constants():
    assert ACTION_DIM == 7, "LIBERO uses 7-DoF actions (6 EEF deltas + gripper)"
    assert NUM_ACTIONS_CHUNK == 8, "OpenVLA-OFT LIBERO uses an 8-step action chunk"
