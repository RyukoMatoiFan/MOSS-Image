"""Expand Comfy INT8 tensorwise / ConvRot checkpoints into dense CPU weights.

Storage decoding follows Comfy-Org/comfy-quants docs/formats/int8_tensorwise.md:
codes represent W H, with per-output-row fp32 scales. The regular normalized
Hadamard is symmetric and self-inverse. This restores the dense quantized model;
it does not reproduce Comfy's runtime activation quantization (W8A8).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file

from .convrot import rotate_hadamard

_DENSE_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _metadata(marker: torch.Tensor, name: str) -> dict:
    if marker.dtype != torch.uint8 or marker.ndim != 1:
        raise ValueError(f"{name}: comfy_quant must be a one-dimensional uint8 JSON marker.")
    try:
        config = json.loads(bytes(marker.cpu().tolist()).decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{name}: malformed comfy_quant JSON.") from exc
    if not isinstance(config, dict) or config.get("format") != "int8_tensorwise":
        raise ValueError(f"{name}: only Comfy int8_tensorwise storage is supported; packed/GPTQ formats are unsupported.")
    if set(config) - {"format", "convrot", "convrot_groupsize", "full_precision_matrix_mult"}:
        raise ValueError(f"{name}: unsupported quantization metadata fields.")
    if not isinstance(config.get("convrot", False), bool):
        raise ValueError(f"{name}: convrot must be a boolean.")
    if "convrot_groupsize" in config and not config.get("convrot", False):
        raise ValueError(f"{name}: rotation group specified without convrot.")
    return config


def _decode(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if dtype not in _DENSE_DTYPES:
        raise ValueError("Dense checkpoint output dtype must be float16, bfloat16, float32 or float64.")
    result = {}
    # Pop consumed source tensors: file loads can release each quantized tensor
    # as its expanded replacement is produced, rather than retaining both states.
    for key in list(state):
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key[:-len(".comfy_quant")]
        config = _metadata(state.pop(key), prefix)
        weight_key, scale_key = prefix + ".weight", prefix + ".weight_scale"
        if weight_key not in state or scale_key not in state:
            raise ValueError(f"{prefix}: quantized weight requires weight, weight_scale and comfy_quant together.")
        weight, scale = state.pop(weight_key), state.pop(scale_key)
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError(f"{prefix}: int8_tensorwise weight must be int8 [out, in]; packed tensors are unsupported.")
        if scale.dtype != torch.float32 or tuple(scale.shape) != (weight.shape[0], 1):
            raise ValueError(f"{prefix}: weight_scale must be float32 [out, 1].")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{prefix}: weight_scale must be finite and positive.")
        group = 1
        if config.get("convrot", False):
            group = config.get("convrot_groupsize", 256)
            # The Comfy regular Hadamard construction uses Kronecker powers of H4.
            if (isinstance(group, bool) or not isinstance(group, int) or group < 4
                    or group & (group - 1) or (group.bit_length() - 1) % 2):
                raise ValueError(f"{prefix}: Comfy ConvRot group size must be a power of four >= 4.")
            if weight.shape[1] % group:
                raise ValueError(f"{prefix}: input width is not divisible by declared ConvRot group size.")
        # Embedding tables can contain billions of values. Decode rows in small
        # chunks so fp32 rotation temporaries do not expand the whole table.
        dense = torch.empty(weight.shape, device=weight.device, dtype=dtype)
        rows_per_chunk = max(1, (4 * 1024 * 1024) // max(1, weight.shape[1]))
        for first in range(0, weight.shape[0], rows_per_chunk):
            last = first + rows_per_chunk
            decoded = weight[first:last].float() * scale[first:last]
            if group > 1:
                decoded = rotate_hadamard(decoded, group)
            dense[first:last] = decoded.to(dtype=dtype)
            del decoded
        result[weight_key] = dense
        del weight, scale, dense
    for key in list(state):
        value = state.pop(key)
        if key.endswith(".weight_scale") or key.endswith(".input_scale"):
            raise ValueError(f"{key}: unmatched or unsupported quantization side tensor.")
        if key.endswith(".weight") and value.dtype not in _DENSE_DTYPES:
            raise ValueError(f"{key}: integer/packed/float8 weights need a supported comfy_quant marker.")
        if value.is_floating_point():
            if value.dtype not in _DENSE_DTYPES:
                raise ValueError(f"{key}: unsupported floating-point storage format.")
            value = value.to(dtype=dtype)
        result[key] = value
    return result


def decode_comfy_state(
    state: Mapping[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Decode supported markers without mutating the caller's mapping."""
    return _decode(dict(state), dtype)


def load_comfy_state(
    path: str | Path, dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load a safetensors checkpoint on CPU and expand quantized rows to dense."""
    return _decode(load_file(str(path), device="cpu"), dtype)
