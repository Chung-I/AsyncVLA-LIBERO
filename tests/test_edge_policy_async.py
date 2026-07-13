from experiments.robot.libero.edge_policy import _should_refresh


def test_refresh_cadence():
    # N=1: refresh every step. N=4: steps 0,4,8 refresh; 1,2,3,5 do not.
    assert [_should_refresh(s, 1) for s in range(5)] == [True, True, True, True, True]
    assert [_should_refresh(s, 4) for s in range(8)] == [
        True, False, False, False, True, False, False, False,
    ]
