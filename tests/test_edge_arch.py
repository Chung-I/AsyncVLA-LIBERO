"""CPU-only unit tests for `experiments.robot.libero.edge_arch`.

No GPU and no 7B base model required: these tests only exercise the small trainable
edge adapter (`Edge_adapter_manip`) + action-token projector (`Proj_Actiontokens`), built
directly on `device="cpu"`.
"""

import pytest
import torch

from experiments.robot.libero.edge_arch import EdgeArch, build_edge_and_proj, load_edge_arch, save_edge_arch


def test_load_edge_arch_missing_file_returns_defaults(tmp_path):
    """No `edge_arch.json` next to the checkpoint -> backward-compat default (512/2/2/4),
    matching every Phase-1 checkpoint trained before this module existed."""
    ckpt_path = tmp_path / "shead--50000_checkpoint.pt"
    arch = load_edge_arch(ckpt_path)
    assert arch == EdgeArch()
    assert arch == EdgeArch(
        obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=2, mha_ff_dim_factor=4
    )


def test_save_then_load_edge_arch_round_trips(tmp_path):
    written = EdgeArch(
        obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4
    )
    save_edge_arch(tmp_path, written)

    ckpt_path = tmp_path / "shead--50000_checkpoint.pt"
    loaded = load_edge_arch(ckpt_path)

    assert loaded == written


def test_checkpoint_round_trip_at_nondefault_arch(tmp_path):
    """The important one: build at 1024/4/4, save state_dicts + edge_arch.json, then
    reload the arch from disk, rebuild, and strict-load -- must succeed with no shape
    mismatch, proving `edge_arch.json` is sufficient to reconstruct a loadable edge."""
    arch = EdgeArch(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    device = torch.device("cpu")
    llm_dim = 4096

    edge, proj = build_edge_and_proj(llm_dim, device, arch)

    torch.save(edge.state_dict(), tmp_path / "shead--50000_checkpoint.pt")
    torch.save(proj.state_dict(), tmp_path / "proj--50000_checkpoint.pt")
    save_edge_arch(tmp_path, arch)

    resolved_arch = load_edge_arch(tmp_path / "shead--50000_checkpoint.pt")
    assert resolved_arch == arch

    edge2, proj2 = build_edge_and_proj(llm_dim, device, resolved_arch)
    edge2.load_state_dict(torch.load(tmp_path / "shead--50000_checkpoint.pt", map_location="cpu"), strict=True)
    proj2.load_state_dict(torch.load(tmp_path / "proj--50000_checkpoint.pt", map_location="cpu"), strict=True)


def test_loading_nondefault_state_dict_into_default_arch_raises(tmp_path):
    """Negative check: a 1024/4/4 `shead` state_dict loaded into a DEFAULT (512/2/2) edge
    must RAISE -- proves the persisted arch is what makes eval correct (heads/layers are
    not recoverable from tensor shapes alone, so silently using the wrong default would be
    a much worse failure mode than a loud shape-mismatch error)."""
    trained_arch = EdgeArch(
        obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4
    )
    device = torch.device("cpu")
    llm_dim = 4096

    edge, _ = build_edge_and_proj(llm_dim, device, trained_arch)
    shead_state_dict = edge.state_dict()

    default_edge, _ = build_edge_and_proj(llm_dim, device, EdgeArch())
    with pytest.raises(RuntimeError):
        default_edge.load_state_dict(shead_state_dict, strict=True)
