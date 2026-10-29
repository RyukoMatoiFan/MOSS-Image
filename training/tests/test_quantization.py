"""Numerical and storage checks for frozen int8 adapter bases."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from training.lora import LoRALinear
from training.quantization import FrozenInt8Linear, quantize_frozen_linears


def test_forward_backward_and_saved_storage():
    torch.manual_seed(7)
    base = nn.Linear(64, 32).requires_grad_(False)
    base.weight[0].zero_()
    quantized = FrozenInt8Linear(base)
    tokens = torch.randn(2, 3, 64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_(True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        actual = quantized(tokens)
    expected = F.linear(reference_tokens, quantized.weight, quantized.bias)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    torch.testing.assert_close(actual, base(tokens.detach()), atol=0.015, rtol=0.02)
    assert len(saved) == 2
    assert saved[0].dtype == torch.int8
    assert saved[1].shape == (32,)
    assert not list(quantized.parameters())
    assert all(tensor.shape != base.weight.shape or tensor.dtype == torch.int8 for tensor in quantized.buffers())


def test_conversion_preserves_adapters_and_gradients():
    base = nn.Linear(16, 8).requires_grad_(False)
    model = nn.Sequential(LoRALinear(base, rank=3, alpha=3))
    model[0].lora_B.data.normal_()
    adapter_ids = [id(model[0].lora_A), id(model[0].lora_B)]
    assert quantize_frozen_linears(model) == ["0.base"]
    assert isinstance(model[0].base, FrozenInt8Linear)
    assert [id(model[0].lora_A), id(model[0].lora_B)] == adapter_ids
    assert model[0].lora_A.dtype == torch.float32
    tokens = torch.randn(2, 16, requires_grad=True)
    model(tokens).square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert model[0].lora_A.grad is not None and model[0].lora_B.grad is not None


def test_dtype_moves_and_state_roundtrip():
    base = nn.Linear(16, 8).requires_grad_(False)
    quantized = FrozenInt8Linear(base)
    original_scale = quantized.weight_scale.clone()
    quantized.to(dtype=torch.bfloat16)
    assert quantized.compute_dtype == torch.bfloat16
    assert quantized.weight_int8.dtype == torch.int8
    torch.testing.assert_close(quantized.weight_scale, original_scale, rtol=0, atol=0)
    tokens = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
    quantized(tokens).float().sum().backward()
    assert tokens.grad.dtype == torch.bfloat16
    restored = FrozenInt8Linear(base).to(dtype=torch.bfloat16)
    restored.load_state_dict(quantized.state_dict())
    torch.testing.assert_close(restored(tokens), quantized(tokens))


def test_trainable_base_is_rejected_and_skipped():
    with pytest.raises(ValueError, match="Freeze"):
        FrozenInt8Linear(nn.Linear(4, 4))
    model = nn.Sequential(nn.Linear(4, 4))
    assert quantize_frozen_linears(model) == []


def test_autocast_input_backward_matches_float_linear():
    quantized = FrozenInt8Linear(nn.Linear(16, 8).requires_grad_(False))
    tokens = torch.randn(2, 16, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = quantized(tokens)
        expected = F.linear(reference, quantized.weight, quantized.bias)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tokens.grad, reference.grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA block transfers")
@pytest.mark.parametrize("precision", ["no", "bf16"])
@pytest.mark.parametrize("quantization", ["int8", "int8_convrot"])
def test_int8_lora_edit_offload_gradients_and_optimizer_update(precision, quantization):
    """Quantized buffers survive functional block copies and checkpoint recomputation."""
    import copy

    from training.lora import inject_lora
    from training.memory import place_model
    from training.model import MOSSImageTransformer

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    torch.manual_seed(29)
    resident = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=2,
        text_dim=16, freq_dim=8, latent_channels=4, gradient_checkpointing=True,
    ).train()
    inject_lora(resident, rank=3, alpha=6, targets="blocks_and_io")
    with torch.no_grad():
        for module in resident.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.03)
    if quantization == "int8":
        assert quantize_frozen_linears(resident)
    else:
        from training.convrot import quantize_frozen_convrot_linears

        assert quantize_frozen_convrot_linears(resident)
    swapped = copy.deepcopy(resident)
    device = torch.device("cuda")
    place_model(resident, device, blocks_to_swap=0)
    place_model(swapped, device, blocks_to_swap=2)
    target = torch.randn(1, 4, 4, 4, device=device)
    context = torch.randn(1, 3, 16, device=device)
    references = [torch.randn(1, 4, 2, 4, device=device)]
    trainables = [{name: value for name, value in model.named_parameters() if value.requires_grad}
                  for model in (resident, swapped)]
    initial = {name: value.detach().cpu().clone() for name, value in trainables[0].items()}
    outputs = []
    for model, parameters in zip((resident, swapped), trainables):
        optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
        # Accumulate across two graphs, exercising checkpoint replay and CPU master grads.
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model(target, torch.tensor([500.], device=device), context,
                               ref_hidden_states=references)
                loss = output.float().square().mean()
            outputs.append(output.detach().cpu())
            loss.backward()
        assert all(value.dtype == torch.float32 for value in parameters.values())
        assert all(value.grad is not None and torch.isfinite(value.grad).all()
                   for value in parameters.values())
        assert any(value.grad.abs().max() > 0 for name, value in parameters.items() if name.endswith("lora_A"))
        optimizer.step()
    torch.testing.assert_close(outputs[0], outputs[2], atol=2e-5, rtol=2e-3)
    for name, first in trainables[0].items():
        second = trainables[1][name]
        torch.testing.assert_close(first.grad.cpu(), second.grad.cpu(), atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(first.detach().cpu(), second.detach().cpu(), atol=2e-5, rtol=2e-3)
    assert any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in trainables[0].items())
    assert all(value.device.type == "cpu" for value in swapped.layers.parameters())
    assert all(value.device.type == "cpu" for value in swapped.layers.buffers())
