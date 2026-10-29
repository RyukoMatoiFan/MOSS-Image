"""Checkpoint loading materializes CPU weights without random parameter storage."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from training import weights
from training.lora import LoRALinear
from training.model import MOSSImageTransformer


def _model():
    return MOSSImageTransformer(
        dim=48, ffn_dim=80, num_heads=3, num_layers=3,
        text_dim=20, freq_dim=12, latent_channels=8, rope_base=1234.0,
    )


@pytest.mark.parametrize("prefix", ["", "transformer."])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_meta_loader_restores_bfloat_checkpoint_on_cpu(tmp_path, monkeypatch, prefix, dtype):
    original = _model().to(dtype=torch.bfloat16).eval()
    config = {name: getattr(original, name) for name in (
        "dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim",
        "latent_channels", "rope_base",
    )}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {prefix + key: value for key, value in original.state_dict().items() if key != "rope.bands"}
    save_file(state, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    initialized_on = []

    def construct(**kwargs):
        result = MOSSImageTransformer(**kwargs)
        initialized_on.extend(parameter.device.type for parameter in result.parameters())
        return result

    monkeypatch.setattr(weights, "MOSSImageTransformer", construct)
    loaded = weights.load_transformer(tmp_path, gradient_checkpointing=True, dtype=dtype)
    assert initialized_on and set(initialized_on) == {"meta"}
    assert loaded.gradient_checkpointing and not loaded.training
    assert loaded.rope.bands.device.type == "cpu"
    assert torch.equal(loaded.rope.bands, original.rope.default_bands(torch.device("cpu")))
    assert all(tensor.device.type == "cpu" for tensor in loaded.state_dict().values())
    assert all(parameter.dtype == dtype for parameter in loaded.parameters())
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, original.state_dict()[name].to(dtype))
    inputs = (torch.randn(1, 8, 2, 3, dtype=dtype), torch.tensor([500.0]), torch.randn(1, 4, 20, dtype=dtype))
    assert loaded(*inputs).shape == inputs[0].shape


def test_meta_loader_rejects_missing_parameter(tmp_path):
    state = _model().state_dict()
    del state["video_head_modulation.bias"]
    save_file(state, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    with pytest.raises(ValueError, match="missing transformer weights.*video_head_modulation.bias"):
        weights.load_transformer(tmp_path)


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
@pytest.mark.parametrize("fused", [False, True])
def test_comfy_single_file_without_config(tmp_path, prefix, fused):
    original = _model()
    checkpoint = tmp_path / "moss-dit.safetensors"
    state = original.state_dict()
    if fused:
        for index in range(original.num_layers):
            stem = f"layers.{index}.proj."
            state[stem + "qkv.weight"] = torch.cat([state.pop(stem + key + ".weight") for key in ("q", "k", "v")])
    save_file({prefix + key: value for key, value in state.items()}, str(checkpoint))
    loaded = weights.load_transformer(checkpoint)
    assert (loaded.dim, loaded.num_layers, loaded.text_dim) == (48, 3, 20)
    for name, value in loaded.named_parameters():
        torch.testing.assert_close(value, original.state_dict()[name])


@pytest.mark.parametrize("malformed", ["shape", "duplicate"])
def test_comfy_fused_qkv_rejects_ambiguous_or_invalid_state(tmp_path, malformed):
    state = _model().state_dict()
    key = "layers.0.proj.qkv.weight"
    state[key] = torch.randn(144 if malformed == "duplicate" else 143, 48)
    if malformed == "shape":
        for projection in ("q", "k", "v"):
            del state[f"layers.0.proj.{projection}.weight"]
    checkpoint = tmp_path / "invalid.safetensors"
    save_file(state, str(checkpoint))
    with pytest.raises(ValueError, match="QKV"):
        weights.load_transformer(checkpoint)


def test_comfy_single_file_rejects_packed_weights(tmp_path):
    checkpoint = tmp_path / "packed.safetensors"
    state = _model().state_dict()
    state["layers.0.proj.q.weight"] = state["layers.0.proj.q.weight"].to(torch.int8)
    save_file(state, str(checkpoint))
    with pytest.raises(ValueError, match="requires dense"):
        weights.load_transformer(checkpoint)


@pytest.mark.parametrize("comfy_layout", [False, True])
def test_export_layout_round_trip(tmp_path, comfy_layout):
    original = _model().eval()
    config = {name: getattr(original, name) for name in (
        "dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim",
        "latent_channels", "rope_base",
    )}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    checkpoint = tmp_path / "export.safetensors"
    weights.save_transformer(original, checkpoint, comfy_layout=comfy_layout)
    exported = load_file(str(checkpoint))
    for index in range(original.num_layers):
        stem = f"layers.{index}.proj."
        assert (stem + "qkv.weight" in exported) == comfy_layout
        for projection in ("q", "k", "v"):
            assert (stem + projection + ".weight" in exported) != comfy_layout
        if comfy_layout:
            expected = torch.cat([original.state_dict()[stem + p + ".weight"] for p in ("q", "k", "v")])
            assert torch.equal(exported[stem + "qkv.weight"], expected)
    with safe_open(str(checkpoint), framework="pt") as handle:
        assert handle.metadata()["layout"] == ("comfy" if comfy_layout else "released")
    loaded = weights.load_transformer(checkpoint)
    for name, value in loaded.named_parameters():
        assert torch.equal(value, original.state_dict()[name])
    target, context = torch.randn(1, 8, 2, 3), torch.randn(1, 4, 20)
    torch.testing.assert_close(loaded(target, 500.0, context), original(target, 500.0, context), rtol=0, atol=0)


def test_full_export_rejects_lora_wrappers(tmp_path):
    model = _model()
    model.layers[0].proj["q"] = LoRALinear(model.layers[0].proj["q"], rank=2, alpha=2)
    with pytest.raises(ValueError, match="without LoRA wrappers"):
        weights.save_transformer(model, tmp_path / "invalid.safetensors", comfy_layout=True)
