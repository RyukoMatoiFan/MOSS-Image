"""Block offload strategies for training and inference.

The default synchronous path keeps complete trainable blocks on the CPU and
creates differentiable device copies for each checkpointed invocation. The
asynchronous path is deliberately narrower: frozen block tensors have pinned
CPU masters, LoRA parameters stay on CUDA, and two reusable staging slots
overlap H2D copies with block computation.
"""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


def offloaded_block(layer: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    """Run a synchronously offloaded block from its CPU parameter masters."""

    device = args[0].device
    names, parameters = zip(*layer.named_parameters())
    buffers = dict(layer.named_buffers())

    def run(*inputs: torch.Tensor) -> torch.Tensor:
        values = dict(zip(names, (value.to(device) for value in inputs[: len(names)])))
        values.update({name: value.to(device) for name, value in buffers.items()})
        return functional_call(layer, values, inputs[len(names) :], strict=True)

    if torch.is_grad_enabled():
        return checkpoint(run, *parameters, *args, use_reentrant=False)
    return run(*parameters, *args)


@dataclass
class _BlockTensors:
    cpu_buffer: torch.Tensor
    views: dict[str, "_TensorView"]
    trainable: dict[str, nn.Parameter]


@dataclass(frozen=True)
class _TensorView:
    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class _StagingSlot:
    index: int | None = None
    buffer: torch.Tensor | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    ready: torch.cuda.Event | None = None
    reusable: torch.cuda.Event | None = None
    leased: bool = False
    generation: int = 0


class _RecomputeContext:
    """Reusable context manager for repeated or nested checkpoint recomputations."""

    def __init__(self, phase: ContextVar[bool]) -> None:
        self._phase = phase
        self._tokens = ContextVar("async_block_recompute_tokens", default=())

    def __enter__(self) -> None:
        token = self._phase.set(True)
        self._tokens.set((*self._tokens.get(), token))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        tokens = self._tokens.get()
        if not tokens:
            raise RuntimeError("Async block recompute context exited without being entered.")
        self._tokens.set(tokens[:-1])
        self._phase.reset(tokens[-1])
        return False


def _is_checkpoint_early_stop(error: BaseException) -> bool:
    error_type = type(error)
    return (
        error_type.__name__ == "_StopRecomputationError"
        and error_type.__module__ == "torch.utils.checkpoint"
    )


class AsyncBlockStreamer:
    """H2D-only block streamer with two reusable CUDA staging slots.

    Frozen parameters and buffers are copied from pinned CPU storage. Trainable
    LoRA parameters are read directly from the module on every invocation, so an
    optimizer step cannot leave a stale adapter in the cache.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        device: torch.device,
        first_streamed_block: int,
        blocks: dict[int, _BlockTensors],
    ) -> None:
        self.layers = layers
        self.device = device
        self.first_streamed_block = first_streamed_block
        self.copy_stream = torch.cuda.Stream(device=device)
        self._blocks = blocks
        self._slot_capacity = max(block.cpu_buffer.numel() for block in blocks.values())
        self._slots = [_StagingSlot(), _StagingSlot()]
        self._in_recompute = ContextVar("async_block_recompute", default=False)
        self._recompute_context = _RecomputeContext(self._in_recompute)
        self.peak_resident_blocks = 0
        self.h2d_copy_count = 0
        self._closed = False

    @property
    def resident_block_indices(self) -> tuple[int, ...]:
        """Block indices currently held in the two CUDA staging slots."""

        return tuple(sorted(slot.index for slot in self._slots if slot.index is not None))

    @property
    def closed(self) -> bool:
        """Whether teardown has released this manager's staging resources."""

        return self._closed

    def owned_cuda_buffers(self) -> tuple[torch.Tensor, ...]:
        """Return currently allocated CUDA staging buffers without allocating slots."""

        return tuple(
            slot.buffer
            for slot in self._slots
            if slot.buffer is not None and slot.buffer.device.type == "cuda"
        )

    def _slot_for(self, index: int) -> _StagingSlot | None:
        return next((slot for slot in self._slots if slot.index == index), None)

    def _choose_slot(self, protected: frozenset[int]) -> _StagingSlot:
        empty = next((slot for slot in self._slots if slot.index is None), None)
        if empty is not None:
            return empty
        available = next(
            (slot for slot in self._slots if slot.index not in protected and not slot.leased),
            None,
        )
        if available is None:
            raise RuntimeError("Async block streamer exhausted its two protected staging slots.")
        return available

    def _stage(self, index: int, protected: frozenset[int]) -> _StagingSlot:
        resident = self._slot_for(index)
        if resident is not None:
            return resident

        slot = self._choose_slot(protected)
        block = self._blocks[index]
        with torch.cuda.stream(self.copy_stream):
            if slot.reusable is not None:
                self.copy_stream.wait_event(slot.reusable)
            if slot.buffer is None:
                slot.buffer = torch.empty(
                    self._slot_capacity,
                    dtype=torch.uint8,
                    device=self.device,
                )
            target = slot.buffer[: block.cpu_buffer.numel()]
            target.copy_(block.cpu_buffer, non_blocking=True)
            self.h2d_copy_count += 1
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)

        staged = {
            name: target[view.offset : view.offset + view.nbytes]
            .view(view.dtype)
            .view(view.shape)
            for name, view in block.views.items()
        }

        slot.index = index
        slot.tensors = staged
        slot.ready = ready
        slot.reusable = ready
        slot.leased = False
        slot.generation += 1
        self.peak_resident_blocks = max(self.peak_resident_blocks, len(self.resident_block_indices))
        return slot

    @staticmethod
    def _record_forward_use(slot: _StagingSlot, stream: torch.cuda.Stream) -> None:
        event = torch.cuda.Event()
        event.record(stream)
        slot.reusable = event
        slot.leased = False

    def _record_backward_use(
        self,
        slot: _StagingSlot,
        index: int,
        generation: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> torch.utils.hooks.RemovableHandle:
        differentiable = tuple(tensor for tensor in tensors if tensor.requires_grad)
        if not differentiable:
            raise RuntimeError("Async block checkpoint recomputation requires differentiable tensors.")
        slot.reusable = None
        handle_box: list[torch.utils.hooks.RemovableHandle] = []

        def finished(gradients) -> None:
            try:
                if slot.index != index or slot.generation != generation:
                    raise RuntimeError("Async block staging slot changed before backward completed.")
                stream = torch.cuda.current_stream(self.device)
                event = torch.cuda.Event()
                event.record(stream)
                slot.reusable = event
                slot.leased = False
            finally:
                handle_box[0].remove()

        handle = torch.autograd.graph.register_multi_grad_hook(differentiable, finished, mode="all")
        handle_box.append(handle)
        return handle

    def _run(
        self,
        index: int,
        inputs: tuple[torch.Tensor, ...],
        trainable: dict[str, torch.Tensor],
        completion_tensors: tuple[torch.Tensor, ...],
        forward_prefetch: int | None,
        backward_prefetch: int | None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Async block streamer is closed.")
        recomputing = self._in_recompute.get()
        prefetch = backward_prefetch if recomputing else forward_prefetch
        protected = frozenset(item for item in (index, prefetch) if item is not None)
        slot = self._stage(index, protected)
        if slot.leased:
            raise RuntimeError(f"Async block {index} is already executing in another graph.")
        slot.leased = True
        generation = slot.generation
        compute_stream = torch.cuda.current_stream(self.device)
        backward_handle = None
        try:
            if prefetch is not None:
                self._stage(prefetch, protected)
            if slot.ready is not None:
                compute_stream.wait_event(slot.ready)
            if slot.buffer is not None:
                slot.buffer.record_stream(compute_stream)

            values: dict[str, torch.Tensor] = dict(slot.tensors)
            values.update(trainable)
            if recomputing:
                # Register before functional_call: non-reentrant checkpointing may
                # stop recomputation as soon as its required tensors exist.
                backward_handle = self._record_backward_use(
                    slot,
                    index,
                    generation,
                    completion_tensors,
                )
            output = functional_call(self.layers[index], values, inputs, strict=True)
        except BaseException as error:
            # Non-reentrant checkpointing deliberately interrupts recomputation
            # once its saved tensors are rebuilt. The input-gradient hook still
            # owns the slot until that block's backward completes.
            if not recomputing or not _is_checkpoint_early_stop(error):
                if backward_handle is not None:
                    backward_handle.remove()
                self._record_forward_use(slot, compute_stream)
            raise
        if not recomputing:
            self._record_forward_use(slot, compute_stream)
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
        """Run one streamed block and prefetch in the active traversal direction."""

        if index not in self._blocks:
            raise ValueError(f"Block {index} is not configured for async streaming.")

        if checkpoint_block:
            input_count = len(inputs)
            trainable_names = tuple(self._blocks[index].trainable)
            input_aliases = tuple(
                tensor.view_as(tensor) if tensor.requires_grad else tensor for tensor in inputs
            )
            trainable_aliases = tuple(
                parameter.view_as(parameter)
                for parameter in self._blocks[index].trainable.values()
            )

            def run(*arguments: torch.Tensor) -> torch.Tensor:
                block_inputs = arguments[:input_count]
                adapter_values = dict(zip(trainable_names, arguments[input_count:]))
                completion_tensors = tuple(tensor for tensor in arguments if tensor.requires_grad)
                return self._run(
                    index,
                    block_inputs,
                    adapter_values,
                    completion_tensors,
                    forward_prefetch,
                    backward_prefetch,
                )

            return checkpoint(
                run,
                *input_aliases,
                *trainable_aliases,
                use_reentrant=False,
                context_fn=self._checkpoint_contexts,
            )
        return self._run(
            index,
            inputs,
            self._blocks[index].trainable,
            (),
            forward_prefetch,
            backward_prefetch,
        )

    def close(self) -> None:
        """Finish owned transfers and release cached device copies."""

        if self._closed:
            return
        if any(slot.leased for slot in self._slots):
            raise RuntimeError("Cannot replace async block placement while a block backward is active.")
        self.copy_stream.synchronize()
        for slot in self._slots:
            if slot.reusable is not None:
                slot.reusable.synchronize()
            slot.index = None
            slot.buffer = None
            slot.tensors.clear()
            slot.ready = None
            slot.reusable = None
            slot.leased = False
            slot.generation += 1
        self._blocks.clear()
        self._closed = True


def _is_lora_parameter(name: str) -> bool:
    return name.endswith("lora_A") or name.endswith("lora_B")


def _validate_placement(
    model: nn.Module,
    device: torch.device,
    blocks_to_swap: int,
    block_swap_mode: str,
) -> None:
    if block_swap_mode not in {"sync", "async", "async_full"}:
        raise ValueError(
            "block_swap_mode must be 'sync' or 'async'. Full training additionally accepts 'async_full'."
        )
    if not 0 <= blocks_to_swap <= len(model.layers):
        raise ValueError("blocks_to_swap must be between zero and the number of joint blocks.")
    if block_swap_mode == "sync":
        return
    if device.type != "cuda":
        raise ValueError(f"{block_swap_mode} block swapping requires a CUDA device.")
    boundary = len(model.layers) - blocks_to_swap
    if block_swap_mode == "async_full":
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("async_full requires at least one trainable model parameter.")
        dtypes = {parameter.dtype for parameter in trainable}
        if len(dtypes) != 1 or not dtypes <= {torch.float32, torch.bfloat16}:
            raise ValueError(
                "async_full requires homogeneous FP32 or BF16 trainable model parameters."
            )
        seen: dict[int, str] = {}
        for index, layer in enumerate(model.layers):
            if index < boundary:
                continue
            for name, parameter in layer.named_parameters():
                qualified_name = f"layers.{index}.{name}"
                if not parameter.is_floating_point():
                    raise TypeError(f"async_full cannot stream non-floating parameter {qualified_name}.")
                previous = seen.setdefault(id(parameter), qualified_name)
                if previous != qualified_name:
                    raise ValueError(
                        "async_full does not support parameters shared across streamed blocks "
                        f"({previous} and {qualified_name})."
                    )
        return
    invalid = [
        f"layers.{index}.{name}"
        for index, layer in enumerate(model.layers)
        if index >= boundary
        for name, parameter in layer.named_parameters()
        if parameter.requires_grad and not _is_lora_parameter(name)
    ]
    if invalid:
        example = invalid[0]
        raise ValueError(
            "async block swapping is H2D-only and requires frozen base weights; "
            f"only LoRA adapters may be trainable (found {example})."
        )


def _swap_parameter_storage(parameter: nn.Parameter, moved: torch.Tensor) -> None:
    replacement = nn.Parameter(moved, requires_grad=parameter.requires_grad)
    if parameter.grad is not None:
        replacement.grad = parameter.grad.detach().to(moved.device)
    torch.utils.swap_tensors(parameter, replacement)


def _move_parameter_storage(parameter: nn.Parameter, device: torch.device | str) -> None:
    _swap_parameter_storage(parameter, parameter.detach().to(device))


def _align_bytes(offset: int, alignment: int = 256) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _typed_view(buffer: torch.Tensor, view: _TensorView) -> torch.Tensor:
    return buffer[view.offset : view.offset + view.nbytes].view(view.dtype).view(view.shape)


def _place_async_layer(layer: nn.Module, device: torch.device) -> _BlockTensors:
    parameters = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())
    trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    for parameter in trainable.values():
        _move_parameter_storage(parameter, device)

    frozen = {
        **{name: parameter for name, parameter in parameters.items() if not parameter.requires_grad},
        **buffers,
    }
    views: dict[str, _TensorView] = {}
    offset = 0
    for name, tensor in frozen.items():
        offset = _align_bytes(offset)
        nbytes = tensor.numel() * tensor.element_size()
        views[name] = _TensorView(offset, nbytes, tensor.shape, tensor.dtype)
        offset += nbytes
    cpu_buffer = torch.empty(_align_bytes(offset), dtype=torch.uint8, device="cpu", pin_memory=True)

    modules = dict(layer.named_modules())
    for name, source in frozen.items():
        packed = _typed_view(cpu_buffer, views[name])
        packed.copy_(source.detach().to("cpu").contiguous())
        if name in parameters:
            _swap_parameter_storage(parameters[name], packed)
        else:
            prefix, _, local_name = name.rpartition(".")
            modules[prefix]._buffers[local_name] = packed
    return _BlockTensors(cpu_buffer=cpu_buffer, views=views, trainable=trainable)


def place_model(
    model: nn.Module,
    device: torch.device,
    blocks_to_swap: int,
    *,
    block_swap_mode: str = "sync",
) -> None:
    """Place a model for resident, frozen-async, or full-training async execution."""

    device = torch.device(device)
    _validate_placement(
        model,
        device,
        blocks_to_swap,
        block_swap_mode,
    )
    existing_manager = getattr(model, "block_swap_manager", None)
    if existing_manager is not None:
        existing_manager.close()
    model.blocks_to_swap = blocks_to_swap
    model.block_swap_mode = block_swap_mode
    model.block_swap_manager = None

    for name, child in model.named_children():
        if name != "layers":
            child.to(device)
    boundary = len(model.layers) - blocks_to_swap
    if block_swap_mode == "sync":
        for index, layer in enumerate(model.layers):
            layer.to("cpu" if index >= boundary else device)
        return

    if block_swap_mode == "async_full":
        from training.trainable_offload import TrainableBlockStreamer

        for layer in model.layers[:boundary]:
            layer.to(device)
        if blocks_to_swap:
            model.block_swap_manager = TrainableBlockStreamer(
                model.layers,
                device,
                boundary,
                master_dtype=None,
            )
        return

    packed_blocks: dict[int, _BlockTensors] = {}
    for index, layer in enumerate(model.layers):
        if index < boundary:
            layer.to(device)
        else:
            packed_blocks[index] = _place_async_layer(layer, device)
    if blocks_to_swap:
        model.block_swap_manager = AsyncBlockStreamer(
            model.layers,
            device,
            boundary,
            packed_blocks,
        )
