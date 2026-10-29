"""ConvRot basis, frozen quantization and input-gradient checks."""

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from training.convrot import (
    FrozenConvRotLinear,
    effective_group_size,
    quantize_frozen_convrot_linears,
    rotate_hadamard,
)


@pytest.mark.parametrize("group", [1, 2, 4, 8, 64, 256])
def test_rotation_is_self_inverse_and_preserves_linear(group):
    torch.manual_seed(32)
    x = torch.randn(2, 3, group * 2, dtype=torch.float64)
    weight = torch.randn(5, group * 2, dtype=torch.float64)
    rotated = rotate_hadamard(x, group)
    torch.testing.assert_close(rotate_hadamard(rotated, group), x)
    torch.testing.assert_close(F.linear(rotated, rotate_hadamard(weight, group)), F.linear(x, weight))


def test_convrot_quantized_output_backward_and_int8_save():
    torch.manual_seed(4)
    base = nn.Linear(256, 32).requires_grad_(False)
    quant = FrozenConvRotLinear(base)
    x = torch.randn(2, 5, 256, requires_grad=True)
    actual = quant(x)
    # Compare against the mathematically equivalent unrotated decoded weight.
    expected = F.linear(x, quant.weight, quant.bias)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert (actual - base(x)).square().mean().sqrt() < 0.01
    gradient = torch.randn_like(actual)
    actual_grad, = torch.autograd.grad(actual, x, gradient)
    reference_grad, = torch.autograd.grad(expected, x, gradient)
    torch.testing.assert_close(actual_grad, reference_grad, atol=2e-6, rtol=2e-5)
    assert not list(quant.parameters())
    buffer = io.BytesIO()
    torch.save(quant.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    assert state["weight_int8"].dtype == torch.int8
    assert state["rotation_group_size"].item() == 256
    clone = FrozenConvRotLinear(base)
    clone.load_state_dict(state)
    torch.testing.assert_close(clone(x), actual)


def test_group_fallback_and_conversion_keeps_trainable_linears():
    assert effective_group_size(48, 256) == 16
    assert effective_group_size(17, 256) == 1
    model = nn.Sequential(nn.Linear(48, 8).requires_grad_(False), nn.Linear(8, 3))
    assert quantize_frozen_convrot_linears(model) == ["0"]
    assert model[0].group_size == 16
    assert isinstance(model[1], nn.Linear)
    model.to(dtype=torch.bfloat16)
    assert model[0].compute_dtype == torch.bfloat16
    assert model[0].weight_scale.dtype == torch.float32
    assert model[0].weight_int8.dtype == torch.int8


def test_trainable_base_and_invalid_group_rejected():
    with pytest.raises(ValueError, match="Freeze"):
        FrozenConvRotLinear(nn.Linear(16, 4))
    with pytest.raises(ValueError, match="positive"):
        effective_group_size(16, 0)


def test_regular_h4_convention_and_mismatched_checkpoint_rejected():
    regular = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                            [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
    torch.testing.assert_close(rotate_hadamard(torch.eye(4), 4), regular)
    base = nn.Linear(16, 4).requires_grad_(False)
    source = FrozenConvRotLinear(base, 16)
    target = FrozenConvRotLinear(base, 4)
    with pytest.raises(RuntimeError, match="group size differs"):
        target.load_state_dict(source.state_dict())
