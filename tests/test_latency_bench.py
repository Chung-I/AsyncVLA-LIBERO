import pytest
pytestmark = pytest.mark.slow


def test_benchmark_returns_positive_times():
    from experiments.robot.libero.latency_bench import benchmark
    out = benchmark(n_warmup=2, n_iter=5)
    assert out["t_base_ms"] > 0.0 and out["t_edge_ms"] > 0.0
    # The edge is ~100x smaller than the 7B base -> must be much faster.
    assert out["t_edge_ms"] < out["t_base_ms"]
