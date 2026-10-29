"""Scoped saved-activation offload for CUDA training.

The hooks in this module are deliberately local to a model forward. CPU runs are
an exact no-op and do not query or initialize CUDA.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch
from torch import nn


@dataclass
class _OffloadedTensor:
    cpu: torch.Tensor
    device: torch.device
    ready: torch.cuda.Event
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    payload_shape: tuple[int, ...]


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    device = tensor.device
    return device.type, device.index, tensor.untyped_storage().data_ptr()


def _offload_view(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]] | None:
    """Return a non-overlapping view suitable for a dense CPU payload.

    Expanded dimensions are narrowed to one element before copying. Other
    overlapping layouts stay on device because a logical copy cannot recreate
    their aliasing safely.
    """

    shape = list(tensor.shape)
    source = tensor
    for dimension, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        if stride < 0:
            return None
        if stride == 0 and size > 1:
            source = source.narrow(dimension, 0, 1)
            shape[dimension] = 1

    # Check that the remaining logical elements occupy unique storage offsets.
    # Size-one dimensions do not contribute and may keep arbitrary strides.
    extent = 1
    dimensions = sorted(
        (stride, size)
        for size, stride in zip(shape, tensor.stride())
        if size > 1
    )
    for stride, size in dimensions:
        if stride < extent:
            return None
        extent += (size - 1) * stride
    return source, tuple(shape)


def _restore_layout(
    payload: torch.Tensor,
    device: torch.device,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    payload_shape: tuple[int, ...],
    *,
    non_blocking: bool,
) -> torch.Tensor:
    """Copy a dense payload into the original padded or expanded layout."""

    restored = torch.empty_strided(payload_shape, stride, dtype=payload.dtype, device=device)
    restored.copy_(payload, non_blocking=non_blocking)
    return restored.as_strided(shape, stride)


def activation_cpu_offload(
    model: nn.Module,
    enabled: bool,
    device: torch.device,
) -> ContextManager[object]:
    """Offload tensors saved by autograd to pinned CPU memory.

    Parameter-backed tensors remain resident. This avoids copying full weights
    when an autograd formula saves a parameter or one of its views. CUDA events
    order asynchronous device-to-host copies before backward restores a tensor.
    """

    if not enabled or device.type != "cuda" or not torch.is_grad_enabled():
        return nullcontext()

    resident_storages = {
        _storage_key(tensor)
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type == "cuda" and tensor.untyped_storage().nbytes()
    }

    def is_weight_backed(tensor: torch.Tensor) -> bool:
        key = _storage_key(tensor)
        if key in resident_storages:
            return True
        manager = getattr(model, "block_swap_manager", None)
        owned = getattr(manager, "owned_cuda_buffers", None)
        if not callable(owned):
            return False
        # Streamers own at most two staging rings. Querying them at pack time
        # also catches buffers allocated lazily after this context was entered.
        return any(
            buffer.device.type == "cuda" and _storage_key(buffer) == key
            for buffer in owned()
        )

    def pack(tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        if tensor.device.type != "cuda" or is_weight_backed(tensor):
            return tensor
        view = _offload_view(tensor)
        if view is None:
            return tensor
        source, payload_shape = view
        cpu = torch.empty(payload_shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        cpu.copy_(source, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
        return _OffloadedTensor(
            cpu,
            tensor.device,
            ready,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            payload_shape,
        )

    def unpack(saved: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
        if isinstance(saved, torch.Tensor):
            return saved
        torch.cuda.current_stream(saved.device).wait_event(saved.ready)
        return _restore_layout(
            saved.cpu,
            saved.device,
            saved.shape,
            saved.stride,
            saved.payload_shape,
            non_blocking=True,
        )

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
