"""Optional model fast paths with unchanged default behavior.

This module contains no CUDA initialization and does not mutate global backend
flags. The training and sampling entry points opt into a backend, while the model
can continue to use PyTorch SDPA and eager blocks by default.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from training.lora import LoRALinear

_ATTENTION_BACKENDS = {"auto", "sdpa", "sdpa_cudnn", "flash_attn_2"}
_LORA_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _normalize_attention_backend(backend: str) -> str:
    value = str(backend).strip().lower().replace("-", "_")
    if value not in _ATTENTION_BACKENDS:
        choices = ", ".join(sorted(_ATTENTION_BACKENDS))
        raise ValueError(f"attention backend must be one of: {choices}; got {backend!r}.")
    return value


def _normalize_lora_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        dtype = value
    else:
        dtype = _LORA_DTYPES.get(str(value).strip().lower())
        if dtype is None:
            choices = ", ".join(sorted(_LORA_DTYPES))
            raise ValueError(f"LoRA compute dtype must be a torch dtype or one of: {choices}; got {value!r}.")
    if dtype not in {None, torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("LoRA compute dtype must be float16, bfloat16, float32, or None.")
    return dtype


@lru_cache(maxsize=1)
def _load_flash_attn_func() -> Callable[..., torch.Tensor]:
    """Import FlashAttention lazily, without touching CUDA during configuration."""

    try:
        package = importlib.import_module("flash_attn")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "flash_attn_2 was requested, but flash-attn could not be imported. "
            "Install a wheel matching this Python, PyTorch, and CUDA build; for the "
            "current Windows training environment that means a cp310/torch2.9/cu126 wheel."
        ) from exc
    function = getattr(package, "flash_attn_func", None)
    if not callable(function):
        raise RuntimeError("The installed flash-attn package does not expose flash_attn_func.")
    return function


@lru_cache(maxsize=1)
def _load_flash_attn_varlen_func() -> Callable[..., torch.Tensor]:
    """Return FlashAttention's packed variable-length kernel."""

    try:
        package = importlib.import_module("flash_attn")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "flash_attn_2 was requested, but flash-attn could not be imported. "
            "Install a wheel matching this Python, PyTorch, and CUDA environment."
        ) from exc
    function = getattr(package, "flash_attn_varlen_func", None)
    if not callable(function):
        raise RuntimeError("The installed flash-attn package does not expose flash_attn_varlen_func.")
    return function


def validate_attention_backend(
    backend: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> str:
    """Validate backend availability and optional tensor requirements.

    Passing no device is a package-only preflight suitable before the model is
    loaded onto a GPU. It never allocates a CUDA tensor or queries a CUDA device.
    """

    selected = _normalize_attention_backend(backend)
    target = torch.device(device) if device is not None else None
    if selected == "flash_attn_2":
        _load_flash_attn_func()
        if target is not None and target.type != "cuda":
            raise ValueError("flash_attn_2 requires CUDA tensors.")
        if dtype is not None and dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("flash_attn_2 requires float16 or bfloat16 query/key/value tensors.")
    elif selected == "sdpa_cudnn" and target is not None and target.type != "cuda":
        raise ValueError("sdpa_cudnn requires CUDA tensors and an enabled cuDNN attention kernel.")
    return selected


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention query, key, and value must be [batch, heads, sequence, head_dim].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("attention query, key, and value must have identical shapes.")
    if query.device != key.device or query.device != value.device:
        raise ValueError("attention query, key, and value must be on the same device.")


def scaled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Run unmasked, non-causal attention on ``[B, H, N, D]`` tensors.

    ``auto`` and ``sdpa`` preserve the existing PyTorch SDPA behavior. Explicit
    backends fail rather than silently changing kernels.
    """

    _validate_qkv(query, key, value)
    # Preserve the model's existing SDPA contract: normalized q/k may be FP32,
    # while the projected value determines the attention compute dtype.
    query = query.to(value.dtype)
    key = key.to(value.dtype)
    selected = validate_attention_backend(backend, device=value.device, dtype=value.dtype)
    if selected in {"auto", "sdpa"}:
        return F.scaled_dot_product_attention(query, key, value)
    if selected == "sdpa_cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(query, key, value)

    # Transpose is a view. FlashAttention accepts [B, N, H, D], so avoid a full
    # sequence copy here and transpose the result back to the model's layout.
    flash_attn_func = _load_flash_attn_func()
    output = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        causal=False,
    )
    return output.transpose(1, 2)


def scaled_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Run isolated attention over concatenated variable-length sequences.

    Args:
        query: Packed query ``[1, heads, total_tokens, head_dim]``.
        key: Packed key with the same shape.
        value: Packed value with the same shape.
        cu_seqlens: Int cumulative lengths ``[batch + 1]``, starting at zero.
        backend: Configured attention backend.

    Returns:
        Packed attention output with the same shape as ``value``.
    """

    _validate_qkv(query, key, value)
    query = query.to(value.dtype)
    key = key.to(value.dtype)
    if query.shape[0] != 1:
        raise ValueError("Packed variable-length attention expects a singleton packed batch.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1D tensor with at least two entries.")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != query.shape[2] or bool(torch.any(lengths <= 0)):
        raise ValueError("cu_seqlens must describe positive sequences covering all packed tokens.")

    selected = validate_attention_backend(backend, device=value.device, dtype=value.dtype)
    if selected == "flash_attn_2":
        function = _load_flash_attn_varlen_func()
        cumulative = cu_seqlens.to(device=value.device, dtype=torch.int32)
        max_length = int(lengths.max())
        output = function(
            query.squeeze(0).transpose(0, 1),
            key.squeeze(0).transpose(0, 1),
            value.squeeze(0).transpose(0, 1),
            cumulative,
            cumulative,
            max_length,
            max_length,
            dropout_p=0.0,
            causal=False,
        )
        return output.transpose(0, 1).unsqueeze(0)
    if selected == "sdpa_cudnn":
        raise ValueError(
            "sdpa_cudnn does not support packed block-diagonal attention; "
            "use 'sdpa' or 'flash_attn_2' for packed batching."
        )

    # SDPA's boolean mask uses True for entries that participate in attention.
    # Building labels is linear in the packed token count; the quadratic mask is
    # the portable fallback when a native varlen kernel is unavailable.
    labels = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=value.device),
        lengths.to(device=value.device, dtype=torch.long),
    )
    block_mask = labels[:, None].eq(labels[None, :]).unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=block_mask)


def _version_pair(version: str) -> tuple[int, int] | None:
    numbers: list[int] = []
    for part in version.split("+", 1)[0].split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
        if len(numbers) == 2:
            return numbers[0], numbers[1]
    return None


def validate_compile_environment() -> tuple[str, str]:
    """Check the installed Torch/Triton pair without importing Triton or CUDA.

    Torch 2.9 is the project's validated profile and requires the Triton 3.5
    line. Other Torch releases still require an installed Triton, but remain the
    caller's responsibility until this project validates their pairing.
    """

    distribution_names = ("triton-windows", "triton") if sys.platform == "win32" else ("triton", "triton-windows")
    triton_name = ""
    triton_version = ""
    for distribution in distribution_names:
        try:
            triton_version = metadata.version(distribution)
            triton_name = distribution
            break
        except metadata.PackageNotFoundError:
            continue
    if not triton_version:
        package = "triton-windows>=3.5,<3.6" if sys.platform == "win32" else "triton>=3.5,<3.6"
        raise RuntimeError(
            "Block compilation requires Triton, but no Triton distribution is installed. "
            f"For Torch 2.9 install {package} in this project's environment."
        )

    torch_version = torch.__version__
    if _version_pair(torch_version) == (2, 9) and _version_pair(triton_version) != (3, 5):
        package = "triton-windows" if sys.platform == "win32" else "triton"
        raise RuntimeError(
            f"Block compilation found Torch {torch_version} with {triton_name} {triton_version}. "
            f"Torch 2.9 requires {package}>=3.5,<3.6; install the matching version in this project's "
            "environment before enabling compile_blocks."
        )
    return torch_version, triton_version


@dataclass(frozen=True)
class ModelPerformanceContext:
    """Plain, unregistered runtime state for one configured model."""

    attention_backend: str = "auto"
    compile_mode: str = "default"
    lora_compute_dtype: torch.dtype | None = None
    compiled_block_forward: Callable[..., torch.Tensor] | None = None

    @property
    def blocks_compiled(self) -> bool:
        return self.compiled_block_forward is not None

    def call_block(self, block: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        implementation = getattr(block, "_forward_impl", None)
        if self.compiled_block_forward is None:
            return implementation(*args) if implementation is not None else block(*args)
        return self.compiled_block_forward(block, *args)


_DEFAULT_CONTEXT = ModelPerformanceContext()


def model_performance_context(model: nn.Module) -> ModelPerformanceContext:
    """Return a model's runtime context, or the eager/default context."""

    context = getattr(model, "_moss_performance", None)
    return context if isinstance(context, ModelPerformanceContext) else _DEFAULT_CONTEXT


def call_model_block(model: nn.Module, block: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    """Call a block through the model's one shared compiled function when enabled."""

    return model_performance_context(model).call_block(block, *args)


def configure_model_performance(
    model: nn.Module,
    attention_backend: str = "auto",
    compile_blocks: bool = False,
    compile_mode: str = "default",
    lora_compute_dtype: torch.dtype | str | None = None,
) -> ModelPerformanceContext:
    """Configure attention, repeated-block compilation, and LoRA compute dtype.

    The compiled target is one unbound block ``forward`` shared by every layer.
    Layers are not wrapped, so parameter names and state-dict keys remain stable.
    Checkpointing and offloading stay in the model's outer orchestration.
    """

    selected_backend = validate_attention_backend(attention_backend)
    adapter_dtype = _normalize_lora_dtype(lora_compute_dtype)
    mode = str(compile_mode).strip()
    if not mode:
        raise ValueError("compile_mode must be a non-empty torch.compile mode.")
    if mode in {"reduce-overhead", "max-autotune"}:
        raise ValueError(
            f"compile_mode={mode!r} is disabled because it enables CUDA graphs, which can retain "
            "large per-block copies; use 'default' or a no-CUDA-graphs mode."
        )

    layers = getattr(model, "layers", ())
    compiled_forward: Callable[..., torch.Tensor] | None = None
    if compile_blocks:
        if not layers:
            raise ValueError("Block compilation requires model.layers with at least one block.")
        block_type = type(layers[0])
        if any(type(layer) is not block_type for layer in layers):
            raise ValueError("All model.layers entries must have the same concrete type for shared compilation.")
        validate_compile_environment()
        block_forward = getattr(block_type, "_forward_impl", block_type.forward)
        compiled_forward = torch.compile(block_forward, mode=mode)

    for module in model.modules():
        if hasattr(module, "heads"):
            module.attention_backend = selected_backend
        if isinstance(module, LoRALinear):
            module.lora_compute_dtype = adapter_dtype

    context = ModelPerformanceContext(
        attention_backend=selected_backend,
        compile_mode=mode,
        lora_compute_dtype=adapter_dtype,
        compiled_block_forward=compiled_forward,
    )
    # Bypass nn.Module.__setattr__: the context and compiled callable are runtime
    # helpers, not registered children that should alter state_dict/module names.
    object.__setattr__(model, "_moss_performance", context)
    for layer in layers:
        object.__setattr__(layer, "_moss_performance", context)
    return context
