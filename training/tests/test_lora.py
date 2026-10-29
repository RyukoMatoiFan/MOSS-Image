"""LoRA identity, key layout, and a tiny training step."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from training.config import TrainConfig
from training.lora import LoRALinear, inject_lora, load_lora, lora_spec, save_lora
from training.model import MOSSImageTransformer
from training.train import _note_resumed_alpha, smoke


def _tiny() -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=2, text_dim=16, freq_dim=8, latent_channels=4,
    )


def test_fresh_lora_matches_the_base_and_saves_comfy_keys(tmp_path) -> None:
    """Zero B leaves the forward unchanged, and the file uses diffusion_model keys."""

    torch.manual_seed(0)
    model = _tiny().eval()
    target = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 3, 16)
    timestep = torch.tensor([100.0])
    before = model(target, timestep, context)
    inject_lora(model, rank=2, alpha=4.0)
    after = model(target, timestep, context)
    assert torch.allclose(before, after, atol=1e-5)
    path = tmp_path / "lora.safetensors"
    save_lora(model, path)
    saved = {
        name: module.lora_A.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.data.zero_()
    load_lora(model, path)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            assert torch.equal(module.lora_A, saved[name])
    rank, alpha, targets = lora_spec(path)
    assert (rank, alpha, targets) == (2, 4.0, "blocks")


def test_load_lora_applies_saved_alpha(tmp_path) -> None:
    """Reloading rank 4 and alpha 8 sets the forward scale to 2."""

    torch.manual_seed(1)
    model = _tiny().eval()
    inject_lora(model, rank=4, alpha=8.0)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.normal_()
                module.lora_B.normal_()
    path = tmp_path / "lora.safetensors"
    save_lora(model, path)
    fresh = _tiny().eval()
    inject_lora(fresh, rank=4, alpha=4.0)
    load_lora(fresh, path)
    module = fresh.layers[0].proj["q"]
    probe = torch.randn(1, module.base.in_features)
    base = torch.nn.functional.linear(probe, module.base.weight, module.base.bias)
    update = torch.nn.functional.linear(torch.nn.functional.linear(probe, module.lora_A), module.lora_B)
    ratio = ((module(probe) - base) / update).mean()
    assert module.scale == 2.0
    assert torch.allclose(ratio, torch.tensor(2.0), atol=1e-5)


def test_rejected_lora_leaves_the_model_unchanged(tmp_path) -> None:
    """A bad file raises before any adapter weight or scale is copied."""

    torch.manual_seed(0)
    source = _tiny()
    inject_lora(source, rank=4, alpha=8.0)
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.normal_()
                module.lora_B.normal_()
    path = tmp_path / "lora.safetensors"
    save_lora(source, path)
    saved = load_file(str(path))

    def assert_unchanged(mutated: dict[str, torch.Tensor], text: str) -> None:
        bad = tmp_path / "bad.safetensors"
        save_file(mutated, str(bad))
        fresh = _tiny()
        inject_lora(fresh, rank=4, alpha=4.0)
        before = {name: parameter.detach().clone() for name, parameter in fresh.named_parameters()}
        scales = [module.scale for module in fresh.modules() if isinstance(module, LoRALinear)]
        try:
            load_lora(fresh, bad)
        except ValueError as exc:
            assert text in str(exc)
        else:
            raise AssertionError(text)
        for name, parameter in fresh.named_parameters():
            assert torch.equal(parameter, before[name])
        assert scales == [module.scale for module in fresh.modules() if isinstance(module, LoRALinear)]

    only_a = {key: value for key, value in saved.items() if not key.endswith(".lora_B.weight")}
    assert_unchanged(only_a, "missing modules")
    extra = dict(saved)
    extra["diffusion_model.layers.9.proj.q.lora_A.weight"] = torch.zeros(4, 32)
    assert_unchanged(extra, "does not match an injected module")
    bad_alpha = dict(saved)
    bad_alpha["diffusion_model.final_linear_video.alpha"] = torch.tensor(2.0)
    assert_unchanged(bad_alpha, "does not match an injected module")
    fresh = _tiny()
    inject_lora(fresh, rank=2, alpha=4.0)
    before = {name: parameter.detach().clone() for name, parameter in fresh.named_parameters()}
    try:
        load_lora(fresh, path)
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("rank mismatch should fail")
    for name, parameter in fresh.named_parameters():
        assert torch.equal(parameter, before[name])


def test_resumed_alpha_replaces_a_uniform_yaml_value() -> None:
    """One alpha in the loaded adapter is what train_config records."""

    model = _tiny()
    inject_lora(model, rank=2, alpha=4.0)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.alpha = 8.0
    config = TrainConfig(alpha=100.0)
    _note_resumed_alpha(model, config)
    assert config.alpha == 8.0
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    modules[0].alpha = 3.0
    modules[1].alpha = 5.0
    mixed = TrainConfig(alpha=100.0)
    _note_resumed_alpha(model, mixed)
    assert mixed.alpha == 100.0


def test_smoke_lora_and_full_steps_run() -> None:
    """Two adapter steps and one full backward stay finite on CPU."""

    smoke()
