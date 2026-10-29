"""Bidirectional packed block streaming for full-parameter CUDA training.

The packed-ring scheduling is adapted from musubi-tuner/kandinsky6-tuner's
``TrainableBlockRingOffloader`` at commit
502e35625209d8c9f099d36a2055da903405ecf9.  Its README identifies this code
as Apache-2.0 (line 346).  This implementation uses the MOSS block-manager API,
supports non-reentrant checkpoint recomputation, and always parks parameters on
their authoritative pinned CPU storage before serialization or teardown.
"""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _align_bytes(offset: int, alignment: int = 256) -> int:
    return (offset + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class _View:
    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class _Binding:
    module: nn.Module
    local_name: str
    parameter: nn.Parameter | None
    view: _View


@dataclass
class _PackedBlock:
    index: int
    cpu_buffer: torch.Tensor
    bindings: list[_Binding]


@dataclass
class _Slot:
    buffer: torch.Tensor
    index: int | None = None
    ready: torch.cuda.Event | None = None
    reusable: torch.cuda.Event | None = None
    dirty: bool = False
    views: list[torch.Tensor] = field(default_factory=list)


class _RecomputeContext:
    def __init__(self, flag: ContextVar[bool]) -> None:
        self._flag = flag
        self._tokens = ContextVar("trainable_block_recompute_tokens", default=())

    def __enter__(self) -> None:
        token = self._flag.set(True)
        self._tokens.set((*self._tokens.get(), token))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        tokens = self._tokens.get()
        if not tokens:
            raise RuntimeError("Trainable block recompute context exited without entry.")
        self._tokens.set(tokens[:-1])
        self._flag.reset(tokens[-1])
        return False


def _is_checkpoint_early_stop(error: BaseException) -> bool:
    error_type = type(error)
    return (
        error_type.__name__ == "_StopRecomputationError"
        and error_type.__module__ == "torch.utils.checkpoint"
    )


def _typed_view(buffer: torch.Tensor, view: _View) -> torch.Tensor:
    return buffer[view.offset : view.offset + view.nbytes].view(view.dtype).view(view.shape)


class TrainableBlockStreamer:
    """Two-slot H2D/D2H ring for checkpointed full-parameter blocks.

    ``layers[first_streamed_block:]`` are packed into unique, pinned CPU master
    buffers.  Their existing ``Parameter`` objects are retained while storage is
    rebound to one of two reusable CUDA buffers.  Forward traversal prefetches
    upward; checkpoint recomputation prefetches downward.  Once the next block's
    backward begins, the previous block's post-accumulate optimizer hooks have
    completed, so its dirty CUDA payload can be copied back before slot reuse.

    The caller must use a single CUDA process, non-reentrant checkpointing, and
    sequential forward/backward shape groups.  Fused optimizer hooks must update
    a parameter before its block is evicted; pass this streamer to
    ``FusedAdafactor.install_hooks`` so completion is reported. Accumulation
    backwards with stepping disabled remain clean and do not issue D2H. Call
    ``finish_backward`` after
    every backward and ``commit`` before evaluation, state export, or model load.
    ``close`` also commits.  FP32 and BF16 CPU masters are supported.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        device: torch.device | str,
        first_streamed_block: int = 0,
        *,
        master_dtype: torch.dtype | None = None,
        ring_size: int = 2,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Trainable block streaming requires a CUDA device.")
        if ring_size != 2:
            raise ValueError("Trainable block streaming uses exactly two staging slots.")
        if not 0 <= first_streamed_block < len(layers):
            raise ValueError("first_streamed_block must select at least one layer.")
        if master_dtype not in {None, torch.float32, torch.bfloat16}:
            raise TypeError("master_dtype must be torch.float32 or torch.bfloat16.")
        self.layers = layers
        self.device = device
        self.first_streamed_block = first_streamed_block
        self.master_dtype = master_dtype
        self.copy_stream = torch.cuda.Stream(device=device)
        self._in_recompute = ContextVar("trainable_block_recompute", default=False)
        self._recompute_context = _RecomputeContext(self._in_recompute)
        self._blocks: dict[int, _PackedBlock] = {}
        self._parameter_block: dict[int, int] = {}
        self._parameter_name: dict[int, str] = {}
        self._expected_updates: dict[int, set[int]] = {}
        self._updated_parameters: set[int] = set()
        self._failed_parameters: dict[int, str] = {}
        self._step_enabled = True
        self._require_all_updates = True
        self._requires_reload = False
        self._last_backward_index: int | None = None
        self._closed = False
        self.h2d_copy_count = 0
        self.d2h_copy_count = 0
        self.peak_resident_blocks = 0

        seen_parameters: set[int] = set()
        capacity = 0
        for index in range(first_streamed_block, len(layers)):
            block = self._pack_block(index, layers[index], seen_parameters)
            self._blocks[index] = block
            expected: set[int] = set()
            for binding in block.bindings:
                parameter = binding.parameter
                if parameter is None or not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                expected.add(parameter_id)
                self._parameter_block[parameter_id] = index
                self._parameter_name[parameter_id] = f"layers.{index}.{binding.local_name}"
            self._expected_updates[index] = expected
            capacity = max(capacity, block.cpu_buffer.numel())
        self._slots = [
            _Slot(torch.empty(capacity, dtype=torch.uint8, device=device))
            for _ in range(2)
        ]

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def requires_reload(self) -> bool:
        return self._requires_reload

    def owned_cuda_buffers(self) -> tuple[torch.Tensor, ...]:
        """Return the live CUDA staging buffers owned by this streamer."""

        return tuple(slot.buffer for slot in self._slots if slot.buffer.device.type == "cuda")

    @property
    def resident_block_indices(self) -> tuple[int, ...]:
        return tuple(sorted(slot.index for slot in self._slots if slot.index is not None))

    def _pack_block(
        self,
        index: int,
        block: nn.Module,
        seen_parameters: set[int],
    ) -> _PackedBlock:
        sources: list[tuple[nn.Module, str, nn.Parameter | None, torch.Tensor]] = []
        seen_tensors: set[int] = set()
        for module in block.modules():
            for local_name, parameter in module._parameters.items():
                if parameter is None or id(parameter) in seen_tensors:
                    continue
                if id(parameter) in seen_parameters:
                    raise ValueError("Trainable block streaming does not support parameters shared across blocks.")
                if not parameter.is_floating_point():
                    raise TypeError(f"Streamed parameter {local_name} must be floating point.")
                seen_tensors.add(id(parameter))
                seen_parameters.add(id(parameter))
                sources.append((module, local_name, parameter, parameter.detach()))
            for local_name, buffer in module._buffers.items():
                if buffer is None or id(buffer) in seen_tensors:
                    continue
                seen_tensors.add(id(buffer))
                sources.append((module, local_name, None, buffer.detach()))

        views: list[_View] = []
        offset = 0
        converted: list[torch.Tensor] = []
        for _, _, parameter, tensor in sources:
            dtype = self.master_dtype if parameter is not None and self.master_dtype is not None else tensor.dtype
            source = tensor.to(device="cpu", dtype=dtype).contiguous()
            offset = _align_bytes(offset)
            nbytes = source.numel() * source.element_size()
            views.append(_View(offset, nbytes, source.shape, source.dtype))
            converted.append(source)
            offset += nbytes
        cpu_buffer = torch.empty(_align_bytes(offset), dtype=torch.uint8, device="cpu", pin_memory=True)
        bindings: list[_Binding] = []
        for (module, local_name, parameter, _), view, source in zip(sources, views, converted):
            target = _typed_view(cpu_buffer, view)
            target.copy_(source)
            binding = _Binding(module, local_name, parameter, view)
            bindings.append(binding)
            if parameter is not None:
                parameter.data = target
            else:
                module._buffers[local_name] = target
        return _PackedBlock(index, cpu_buffer, bindings)

    @staticmethod
    def _bind(block: _PackedBlock, flat: torch.Tensor) -> list[torch.Tensor]:
        views = [_typed_view(flat, binding.view) for binding in block.bindings]
        for binding, view in zip(block.bindings, views):
            if binding.parameter is not None:
                binding.parameter.data = view
            else:
                binding.module._buffers[binding.local_name] = view
        return views

    def _slot_for(self, index: int) -> _Slot | None:
        return next((slot for slot in self._slots if slot.index == index), None)

    def _choose_slot(self, protected: frozenset[int]) -> _Slot:
        empty = next((slot for slot in self._slots if slot.index is None), None)
        if empty is not None:
            return empty
        slot = next((slot for slot in self._slots if slot.index not in protected), None)
        if slot is None:
            raise RuntimeError("Both trainable staging slots are protected by active blocks.")
        return slot

    def _evict(self, slot: _Slot) -> None:
        if slot.index is None:
            return
        block = self._blocks[slot.index]
        with torch.cuda.stream(self.copy_stream):
            if slot.reusable is not None:
                self.copy_stream.wait_event(slot.reusable)
            if slot.dirty:
                block.cpu_buffer.copy_(slot.buffer[: block.cpu_buffer.numel()], non_blocking=True)
                self.d2h_copy_count += 1
        self._bind(block, block.cpu_buffer)
        slot.index = None
        slot.ready = None
        slot.reusable = None
        slot.dirty = False
        slot.views.clear()

    def _stage(self, index: int, protected: frozenset[int]) -> _Slot:
        resident = self._slot_for(index)
        if resident is not None:
            return resident
        slot = self._choose_slot(protected)
        self._evict(slot)
        block = self._blocks[index]
        target = slot.buffer[: block.cpu_buffer.numel()]
        with torch.cuda.stream(self.copy_stream):
            target.copy_(block.cpu_buffer, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        slot.index = index
        slot.ready = ready
        slot.reusable = ready
        slot.dirty = False
        slot.views = self._bind(block, target)
        self.h2d_copy_count += 1
        self.peak_resident_blocks = max(self.peak_resident_blocks, len(self.resident_block_indices))
        return slot

    def _finish_previous_backward(self) -> None:
        if self._last_backward_index is None:
            return
        index = self._last_backward_index
        slot = self._slot_for(index)
        if slot is None:
            raise RuntimeError("A trainable block was evicted before its backward completed.")
        slot.reusable = torch.cuda.current_stream(self.device).record_event()
        if self._step_enabled:
            failed = self._expected_updates[index] & self._failed_parameters.keys()
            if failed:
                names = sorted(self._failed_parameters[parameter_id] for parameter_id in failed)
                slot.dirty = False
                self._last_backward_index = None
                self._requires_reload = True
                raise RuntimeError(
                    f"Trainable block {index} had a failed fused optimizer update for {names[:3]}."
                )
            missing = self._expected_updates[index] - self._updated_parameters
            if self._require_all_updates and missing:
                names = sorted(self._parameter_name[parameter_id] for parameter_id in missing)
                slot.dirty = False
                self._last_backward_index = None
                self._requires_reload = True
                raise RuntimeError(
                    f"Trainable block {index} backward completed before all parameter updates; "
                    f"missing {names[:3]}."
                )
            slot.dirty = bool(self._expected_updates[index] & self._updated_parameters)
        else:
            slot.dirty = False
        self._last_backward_index = None

    def set_step_enabled(self, enabled: bool) -> None:
        """Select whether this backward is the optimizer-update boundary."""

        if self._last_backward_index is not None:
            raise RuntimeError("Cannot change the trainable streamer boundary during backward.")
        if self._requires_reload:
            raise RuntimeError("Trainable block streamer requires checkpoint reload after a failed step.")
        self._step_enabled = bool(enabled)
        self._updated_parameters.clear()
        self._failed_parameters.clear()

    def set_require_all_updates(self, required: bool) -> None:
        """Configure whether a visited block may contain unused trainable parameters."""

        if self._last_backward_index is not None:
            raise RuntimeError("Cannot change update requirements during backward.")
        self._require_all_updates = bool(required)

    def mark_updated(self, parameter: nn.Parameter) -> None:
        """Record completion of a streamed parameter's fused optimizer hook."""

        if not self._step_enabled:
            return
        parameter_id = id(parameter)
        if parameter_id in self._parameter_block:
            self._updated_parameters.add(parameter_id)

    def mark_failed(self, parameter: nn.Parameter) -> None:
        """Prevent dirty eviction after a diagnosed fused optimizer failure."""

        if not self._step_enabled:
            return
        parameter_id = id(parameter)
        if parameter_id in self._parameter_block:
            self._failed_parameters[parameter_id] = self._parameter_name[parameter_id]

    def mark_reloaded(self) -> None:
        """Clear the failed-step latch after model and optimizer checkpoint load."""

        if self._last_backward_index is not None or self.resident_block_indices:
            raise RuntimeError("Commit the trainable block streamer before marking it reloaded.")
        self._requires_reload = False
        self._updated_parameters.clear()
        self._failed_parameters.clear()

    def _run(
        self,
        index: int,
        inputs: tuple[torch.Tensor, ...],
        forward_prefetch: int | None,
        backward_prefetch: int | None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Trainable block streamer is closed.")
        recomputing = self._in_recompute.get()
        if recomputing:
            self._finish_previous_backward()
        elif self._last_backward_index is not None:
            # A new sequential group started without an explicit drain. Python
            # cannot re-enter forward until the prior backward call returned,
            # so its final optimizer hooks are complete and writeback is safe.
            self._finish_previous_backward()
        prefetch = backward_prefetch if recomputing else forward_prefetch
        protected = frozenset(item for item in (index, prefetch) if item is not None)
        slot = self._stage(index, protected)
        if prefetch is not None:
            if prefetch not in self._blocks:
                raise ValueError(f"Prefetch block {prefetch} is not streamed.")
            self._stage(prefetch, protected)
        compute_stream = torch.cuda.current_stream(self.device)
        if slot.ready is not None:
            compute_stream.wait_event(slot.ready)
        slot.buffer.record_stream(compute_stream)
        try:
            output = self.layers[index](*inputs)
        except BaseException as error:
            if recomputing and _is_checkpoint_early_stop(error):
                self._last_backward_index = index
            else:
                slot.reusable = compute_stream.record_event()
                slot.dirty = recomputing
            raise
        if recomputing:
            self._last_backward_index = index
        else:
            slot.reusable = compute_stream.record_event()
        return output

    def _checkpoint_contexts(self):
        return nullcontext(), self._recompute_context

    def block(
        self,
        index: int,
        *inputs: torch.Tensor,
        forward_prefetch: int | None,
        backward_prefetch: int | None,
        checkpoint_block: bool,
    ) -> torch.Tensor:
        """Run one streamed block using the existing MOSS manager contract."""

        if index not in self._blocks:
            raise ValueError(f"Block {index} is not configured for trainable streaming.")
        if checkpoint_block:
            aliases = tuple(value.view_as(value) if value.requires_grad else value for value in inputs)

            def run(*arguments: torch.Tensor) -> torch.Tensor:
                return self._run(index, arguments, forward_prefetch, backward_prefetch)

            return checkpoint(
                run,
                *aliases,
                use_reentrant=False,
                context_fn=self._checkpoint_contexts,
            )
        return self._run(index, inputs, forward_prefetch, backward_prefetch)

    def finish_backward(self) -> None:
        """Commit the last dirty block after all optimizer hooks have run."""

        if self._closed:
            raise RuntimeError("Trainable block streamer is closed.")
        self._finish_previous_backward()
        self.commit()

    def commit(self) -> None:
        """Write back dirty slots and park every streamed parameter on CPU."""

        if self._closed:
            return
        self._finish_previous_backward()
        for slot in self._slots:
            self._evict(slot)
        self.copy_stream.synchronize()
        self._updated_parameters.clear()
        self._failed_parameters.clear()

    drain = commit

    def close(self) -> None:
        """Commit authoritative weights and release staging buffers and events."""

        if self._closed:
            return
        self.commit()
        for slot in self._slots:
            slot.buffer = torch.empty(0, dtype=torch.uint8, device="cpu")
        self._closed = True

