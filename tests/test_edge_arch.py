"""CPU-only unit tests for `experiments.robot.libero.edge_arch`.

No GPU and no 7B base model required: these tests only exercise the small trainable
edge adapter (`Edge_adapter`) + action-token projector (`Proj_Actiontokens`), built
directly on `device="cpu"`.
"""

import pytest
import torch

from experiments.robot.libero.edge_arch import (
    EdgeArch,
    build_edge_and_proj,
    load_edge_arch,
    save_edge_arch,
    validate_edge_arch,
)


# ==============================
# load_edge_arch: no silent default (the head count cannot be inferred from weights)
# ==============================


def test_load_edge_arch_missing_file_raises(tmp_path):
    """No `edge_arch.json` next to the checkpoint -> loud error, NOT a silent 512/2/2/4
    default. Head count (`mha_num_attention_heads`) leaves no fingerprint in the weights
    (`nn.MultiheadAttention.in_proj_weight` is `[3*embed_dim, embed_dim]`, independent of
    num_heads), so guessing would silently load a checkpoint's weights into the WRONG head
    split rather than fail."""
    ckpt_path = tmp_path / "shead--50000_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        load_edge_arch(ckpt_path)


def test_load_edge_arch_explicit_arch_skips_json(tmp_path):
    """Escape hatch: an explicit `arch` is used as-is and the (missing) json is never
    consulted."""
    ckpt_path = tmp_path / "shead--50000_checkpoint.pt"
    explicit = EdgeArch(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    assert load_edge_arch(ckpt_path, arch=explicit) == explicit


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


# ==============================
# validate_edge_arch: catches what the weights DO reveal (embed_dim, layer count)
# ==============================


def _build_state_dict(arch: EdgeArch):
    edge, _ = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=arch)
    return edge.state_dict()


def test_validate_edge_arch_correct_arch_passes():
    arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=2, mha_ff_dim_factor=4)
    sd = _build_state_dict(arch)
    validate_edge_arch(arch, sd)  # must not raise


def test_validate_edge_arch_wrong_obs_encoding_size_raises():
    trained_arch = EdgeArch(obs_encoding_size=1024, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    sd = _build_state_dict(trained_arch)

    wrong_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    with pytest.raises(ValueError):
        validate_edge_arch(wrong_arch, sd)


def test_validate_edge_arch_wrong_layer_count_raises():
    trained_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=4, mha_ff_dim_factor=4)
    sd = _build_state_dict(trained_arch)

    wrong_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=2, mha_ff_dim_factor=4)
    with pytest.raises(ValueError):
        validate_edge_arch(wrong_arch, sd)


def test_head_mismatch_is_not_visible_in_weight_shapes():
    """The ONE arch field with no fingerprint in the weights: `in_proj_weight`'s shape
    `[3*embed_dim, embed_dim]` does not depend on `num_heads`, so a checkpoint trained at 4
    heads and one trained at 2 heads (same obs_encoding_size, same layer count) produce
    IDENTICAL state_dict shapes -- `validate_edge_arch` legitimately cannot tell them apart
    (that is why `load_edge_arch` refuses to guess instead, see the tests above)."""
    four_head_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=4, mha_num_attention_layers=2, mha_ff_dim_factor=4)
    two_head_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=2, mha_num_attention_layers=2, mha_ff_dim_factor=4)

    four_head_sd = _build_state_dict(four_head_arch)
    # No raise: embed_dim and layer count both match; heads cannot be cross-checked.
    validate_edge_arch(two_head_arch, four_head_sd)


def test_4head_trained_checkpoint_with_missing_json_is_caught_not_silently_defaulted(tmp_path):
    """The real-world danger scenario: a checkpoint trained at 4 heads, 512 width, 2 layers
    -- i.e. matching `EdgeArch()`'s 512/2/2/4 back-compat defaults in EVERY field
    `validate_edge_arch` can check (obs_encoding_size, layer count) -- but with a DIFFERENT
    head count. Before this fix, a missing `edge_arch.json` would silently return
    `EdgeArch()` (2 heads) and the checkpoint would load cleanly with the WRONG head split,
    with no error anywhere. After this fix, the missing json raises instead of guessing."""
    trained_arch = EdgeArch(obs_encoding_size=512, mha_num_attention_heads=4, mha_num_attention_layers=2, mha_ff_dim_factor=4)
    edge, _ = build_edge_and_proj(llm_dim=4096, device=torch.device("cpu"), arch=trained_arch)

    ckpt_path = tmp_path / "shead--50000_checkpoint.pt"
    torch.save(edge.state_dict(), ckpt_path)
    # Deliberately no `save_edge_arch(tmp_path, trained_arch)` call -- simulating the
    # dangerous scenario.

    with pytest.raises(FileNotFoundError):
        load_edge_arch(ckpt_path)
