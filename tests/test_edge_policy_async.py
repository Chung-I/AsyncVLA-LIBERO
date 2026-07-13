import pytest
import torch

from experiments.robot.libero import edge_policy as edge_policy_mod
from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj
from experiments.robot.libero.edge_policy import EdgeEvalConfig, EdgePolicy, _should_refresh


def test_refresh_cadence():
    # N=1: refresh every step. N=4: steps 0,4,8 refresh; 1,2,3,5 do not.
    assert [_should_refresh(s, 1) for s in range(5)] == [True, True, True, True, True]
    assert [_should_refresh(s, 4) for s in range(8)] == [
        True, False, False, False, True, False, False, False,
    ]


# ==============================
# EdgePolicy.__init__'s `edge_arch=` escape hatch (M2): plumbed through to `load_edge_arch`
# so a checkpoint predating `edge_arch.json` can still be evaluated -- without ever falling
# back to a silent 512/2/2/4 default. These tests stub out the heavy frozen-base load
# (`build_frozen_base`, `ActionTokenizer`, `resolve_unnorm_key`) so they run fast on CPU;
# they exercise only the arch-resolution branch of `EdgePolicy.__init__`, which is real (not
# a training-path behavior change -- eval-time construction only).
# ==============================


class _FakeVLA:
    """Stands in for the frozen 7B base: only the attributes `EdgePolicy.__init__` touches
    before it gets to arch resolution (`llm_dim`, `.to(...)`, no `.norm_stats` needed since
    `resolve_unnorm_key` is stubbed too)."""

    llm_dim = 4096

    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    """Only `.tokenizer` is touched (fed to the stubbed `ActionTokenizer`)."""

    tokenizer = object()


def _make_edge_and_proj_checkpoints(tmp_path, arch: EdgeArch):
    edge, proj = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=arch)
    edge_ckpt = tmp_path / "shead--50000_checkpoint.pt"
    proj_ckpt = tmp_path / "proj--50000_checkpoint.pt"
    torch.save(edge.state_dict(), edge_ckpt)
    torch.save(proj.state_dict(), proj_ckpt)
    return str(edge_ckpt), str(proj_ckpt)


@pytest.fixture(autouse=True)
def _stub_heavy_base_load(monkeypatch):
    monkeypatch.setattr(edge_policy_mod, "build_frozen_base", lambda cfg: (_FakeVLA(), _FakeProcessor(), None))
    monkeypatch.setattr(edge_policy_mod, "ActionTokenizer", lambda tokenizer: object())
    monkeypatch.setattr(edge_policy_mod, "resolve_unnorm_key", lambda vla, task_suite_name: "libero_spatial")


def test_edge_policy_missing_edge_arch_json_raises_with_no_explicit_arch(tmp_path):
    """A checkpoint dir with no `edge_arch.json` (every pre-existing Phase-1 checkpoint) and
    no explicit `edge_arch=` passed must still hard-fail -- the loud-failure default is
    preserved, no silent 512/2/2/4 fallback."""
    edge_ckpt, proj_ckpt = _make_edge_and_proj_checkpoints(tmp_path, EdgeArch())
    with pytest.raises(FileNotFoundError):
        EdgePolicy(EdgeEvalConfig(), task_suite_name="libero_spatial", edge_ckpt=edge_ckpt, proj_ckpt=proj_ckpt)


def test_edge_policy_explicit_edge_arch_succeeds_without_json(tmp_path):
    """The escape hatch: passing `edge_arch=` explicitly lets `EdgePolicy` load a checkpoint
    with no `edge_arch.json` next to it -- the four arch fields supplied by the caller (e.g.
    via `run_libero_eval.py`'s `--edge_arch_*` flags) are used as-is."""
    arch = EdgeArch(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    edge_ckpt, proj_ckpt = _make_edge_and_proj_checkpoints(tmp_path, arch)
    policy = EdgePolicy(
        EdgeEvalConfig(), task_suite_name="libero_spatial", edge_ckpt=edge_ckpt, proj_ckpt=proj_ckpt, edge_arch=arch,
    )
    assert policy.edge_arch == arch
