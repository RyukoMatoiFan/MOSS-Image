"""Frozen int8 weight storage for adapter training.

The eager backend dequantizes for each ordinary floating-point GEMM.  The
optional Triton backend decodes weight tiles inside the GEMM instead.  Neither
path caches a full floating-point weight, and the frozen base remains LoRA-only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .quant_kernels import (
    autocast_compute_dtype,
    resolve_quantized_backend,
    triton_int8_linear_backward,
    triton_int8_linear_forward,
    validate_quantized_backend,
)


def _dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return weight.to(dtype) * scale.to(dtype)[:, None]


class _FrozenInt8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, scale, bias, dtype, backend):
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = tokens.dtype
        ctx.weight_dtype = dtype
        selected = resolve_quantized_backend(backend, tokens, weight, scale)
        ctx.backend = selected
        ctx.input_shape = tokens.shape
        if selected == "triton":
            output_dtype = autocast_compute_dtype(tokens, dtype)
            prepared = tokens.to(dtype)
            # Dtype conversion can also make an otherwise strided input contiguous.
            resolve_quantized_backend("triton", prepared, weight, scale)
            result = triton_int8_linear_forward(prepared, weight, scale, bias, output_dtype, dtype)
        else:
            result = F.linear(tokens.to(dtype), _dequantize(weight, scale, dtype), bias)
        ctx.compute_dtype = result.dtype
        return result

    @staticmethod
    def backward(ctx, gradient):
        weight, scale = ctx.saved_tensors
        if ctx.backend == "triton":
            grad_input = triton_int8_linear_backward(
                gradient,
                weight,
                scale,
                ctx.input_shape,
                ctx.input_dtype,
                ctx.weight_dtype,
            )
            return grad_input, None, None, None, None, None
        # Autocast can choose a different output dtype from the stored compute dtype.
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            grad_input = gradient.to(ctx.compute_dtype).matmul(
                _dequantize(weight, scale, ctx.weight_dtype).to(ctx.compute_dtype)
            ).to(ctx.input_dtype)
        return grad_input, None, None, None, None, None


class FrozenInt8Linear(nn.Module):
    """Per-output-channel symmetric int8 linear with gradients only to its input.

    ``weight`` is a compatibility accessor that allocates a transient float tensor.
    Use ``compute_dtype`` for dtype queries and call the module for computation;
    calling ``F.linear`` on ``weight`` bypasses memory-efficient backward.
    """

    def __init__(self, linear: nn.Linear, *, backend: str = "eager") -> None:
        super().__init__()
        if linear.weight.requires_grad or (linear.bias is not None and linear.bias.requires_grad):
            raise ValueError("Freeze the base linear before quantizing it (LoRA training only).")
        if linear.weight.device.type == "meta":
            raise ValueError("Load the base weights before quantizing them.")
        if not torch.isfinite(linear.weight).all():
            raise ValueError("Cannot quantize non-finite linear weights.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self._quantized_backend = validate_quantized_backend(backend)
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer("weight_int8", (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("_dtype_anchor", torch.empty(0, device=weight.device, dtype=linear.weight.dtype), persistent=False)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._dtype_anchor.dtype

    @property
    def quantized_backend(self) -> str:
        return self._quantized_backend

    @quantized_backend.setter
    def quantized_backend(self, backend: str) -> None:
        self._quantized_backend = validate_quantized_backend(backend)

    def set_quantized_backend(self, backend: str) -> FrozenInt8Linear:
        """Select execution for this module and return it for fluent setup."""
        self.quantized_backend = backend
        return self

    @property
    def weight(self) -> torch.Tensor:
        return _dequantize(self.weight_int8, self.weight_scale, self.compute_dtype)

    def _apply(self, fn, recurse=True):
        # Keep scales at fp32 even when model.to(dtype=...) casts other floats.
        scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        self.weight_scale = scale.to(device=self.weight_int8.device, dtype=torch.float32)
        return result

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return _FrozenInt8Linear.apply(
            tokens,
            self.weight_int8,
            self.weight_scale,
            self.bias,
            self.compute_dtype,
            self.quantized_backend,
        )


def configure_quantized_backend(model: nn.Module, backend: str) -> list[str]:
    """Set the backend on every frozen INT8 module and return their paths."""
    selected = validate_quantized_backend(backend)
    names = []
    for name, child in model.named_modules():
        if isinstance(child, FrozenInt8Linear):
            child.quantized_backend = selected
            names.append(name)
    return names


def quantize_frozen_linears(model: nn.Module, *, backend: str = "eager") -> list[str]:
    """Replace frozen Linear children, including bases inside existing LoRA wrappers.

    Call after injecting LoRA so adapter parameters remain floating and trainable.
    Trainable linears are intentionally left alone. Returns replaced module paths.
    """
    validate_quantized_backend(backend)
    names = []
    for name, child in list(model.named_modules()):
        if not name or not isinstance(child, nn.Linear):
            continue
        if any(parameter.requires_grad for parameter in child.parameters()):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, FrozenInt8Linear(child, backend=backend))
        names.append(name)
    return names
