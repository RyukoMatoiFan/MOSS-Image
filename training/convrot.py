"""Frozen INT8 weights in a grouped, normalized Hadamard basis.

This is weight-only ConvRot: activations and GEMMs stay floating point. It is
distinct from ordinary INT8 storage and does not implement a fused W8A8 kernel.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .quant_kernels import (
    autocast_compute_dtype,
    resolve_quantized_backend,
    triton_convrot_backward,
    triton_convrot_forward,
    validate_quantized_backend,
)
from .quantization import FrozenInt8Linear, _FrozenInt8Linear


def effective_group_size(in_features: int, group_size: int = 256) -> int:
    """Largest power of two within the requested limit dividing the input width."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("ConvRot group_size must be a positive integer.")
    if in_features < 1:
        raise ValueError("ConvRot input width must be positive.")
    group = 1 << (group_size.bit_length() - 1)
    while in_features % group:
        group //= 2
    return group


def rotate_hadamard(tokens: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply a symmetric orthonormal transform to groups of the last dimension.

    Powers of four use the regular H4 Kronecker construction used by ConvRot;
    an additional H2 factor handles other powers of two. No dense group matrix
    is allocated, and ordinary autograd propagates the inverse rotation.
    """
    if group_size < 1 or group_size & (group_size - 1):
        raise ValueError("Hadamard group size must be a positive power of two.")
    if tokens.shape[-1] % group_size:
        raise ValueError("Input width must be divisible by Hadamard group size.")
    if group_size == 1:
        return tokens
    shape = tokens.shape
    value = tokens.reshape(-1, group_size)
    stride = 1
    while stride * 4 <= group_size:
        parts = value.reshape(-1, group_size // (4 * stride), 4, stride)
        a, b, c, d = parts.unbind(dim=-2)
        value = torch.stack((a + b + c - d, a + b - c + d,
                             a - b + c + d, -a + b + c + d), dim=-2).reshape(-1, group_size) * 0.5
        stride *= 4
    if stride < group_size:
        parts = value.reshape(-1, 2, stride)
        a, b = parts.unbind(dim=-2)
        value = torch.stack((a + b, a - b), dim=-2).reshape(-1, group_size) / math.sqrt(2)
    return value.reshape(shape)


class _FrozenConvRotTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, scale, bias, dtype, group_size):
        output_dtype = autocast_compute_dtype(tokens, dtype)
        result = triton_convrot_forward(tokens, weight, scale, bias, output_dtype, dtype, group_size)
        ctx.save_for_backward(weight, scale)
        ctx.input_shape = tokens.shape
        ctx.input_dtype = tokens.dtype
        ctx.weight_dtype = dtype
        ctx.group_size = group_size
        return result

    @staticmethod
    def backward(ctx, gradient):
        weight, scale = ctx.saved_tensors
        grad_input = triton_convrot_backward(
            gradient,
            weight,
            scale,
            ctx.input_shape,
            ctx.input_dtype,
            ctx.weight_dtype,
            ctx.group_size,
        )
        return grad_input, None, None, None, None, None


class FrozenConvRotLinear(FrozenInt8Linear):
    """Quantize W H and compute (x H)(W H)^T, with frozen base weights."""

    def __init__(self, linear: nn.Linear, group_size: int = 256, *, backend: str = "eager") -> None:
        # Validate frozen status before creating the temporary rotated linear.
        if any(parameter.requires_grad for parameter in linear.parameters()):
            raise ValueError("Freeze the base linear before ConvRot quantization (LoRA only).")
        if linear.weight.device.type == "meta":
            raise ValueError("Load the base weights before ConvRot quantization.")
        group = effective_group_size(linear.in_features, group_size)
        rotated = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None,
                            device="meta", dtype=torch.float32)
        rotated.weight = nn.Parameter(rotate_hadamard(linear.weight.detach().float(), group), requires_grad=False)
        if linear.bias is not None:
            rotated.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        super().__init__(rotated, backend=backend)
        self._dtype_anchor = torch.empty(0, device=linear.weight.device, dtype=linear.weight.dtype)
        self.group_size = group
        self.register_buffer("rotation_group_size", torch.tensor(group, device=linear.weight.device))

    @property
    def weight(self) -> torch.Tensor:
        """Transient dequantized weight in the original, unrotated basis."""
        return rotate_hadamard(super().weight, self.group_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        selected = resolve_quantized_backend(
            self.quantized_backend,
            tokens,
            self.weight_int8,
            self.weight_scale,
        )
        if selected == "triton":
            return _FrozenConvRotTriton.apply(
                tokens,
                self.weight_int8,
                self.weight_scale,
                self.bias,
                self.compute_dtype,
                self.group_size,
            )
        rotated = rotate_hadamard(tokens, self.group_size)
        return _FrozenInt8Linear.apply(
            rotated,
            self.weight_int8,
            self.weight_scale,
            self.bias,
            self.compute_dtype,
            "eager",
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        stored_group = state_dict.get(prefix + "rotation_group_size")
        if stored_group is not None and int(stored_group.item()) != self.group_size:
            error_msgs.append(f"{prefix}ConvRot group size differs from the checkpoint; "
                              "construct the module with the checkpoint's group size.")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def quantize_frozen_convrot_linears(
    model: nn.Module,
    group_size: int = 256,
    *,
    backend: str = "eager",
) -> list[str]:
    """Replace frozen Linear children after LoRA injection; adapters remain intact."""
    effective_group_size(1, group_size)  # Validate even when there are no candidates.
    validate_quantized_backend(backend)
    names = []
    for name, child in list(model.named_modules()):
        if not name or not isinstance(child, nn.Linear):
            continue
        if any(parameter.requires_grad for parameter in child.parameters()):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, FrozenConvRotLinear(child, group_size, backend=backend))
        names.append(name)
    return names
