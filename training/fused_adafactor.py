"""Memory-bounded Adafactor updates for streamed full-parameter training.

The update and stochastic BF16 copy are adapted from sd-scripts' fused
Adafactor implementation as carried by musubi-tuner/kandinsky6-tuner at
commit 502e35625209d8c9f099d36a2055da903405ecf9.  That source identifies
the code as Apache-2.0 (README, line 346).  This version is a standalone
PyTorch optimizer with explicit accumulation-boundary and checkpoint rules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch import nn

_MOMENT_KEYS = ("exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col")
_OPTIMIZER_KIND = "fused_adafactor_manual_lr_v1"
_GRADIENT_ACCUMULATION_DEVICES = {"device", "cuda", "cpu"}


def stochastic_copy_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy FP32 ``source`` to BF16 ``target`` with stochastic rounding."""

    if target.dtype != torch.bfloat16 or source.dtype != torch.float32:
        raise TypeError("stochastic_copy_ requires a BF16 target and an FP32 source.")
    if target.shape != source.shape or target.device != source.device:
        raise ValueError("stochastic_copy_ requires matching shapes and devices.")
    random_low_bits = torch.randint_like(source, dtype=torch.int32, low=0, high=1 << 16)
    random_low_bits.add_(source.view(torch.int32)).bitwise_and_(-65536)
    target.copy_(random_low_bits.view(torch.float32))


def _split_parameters(
    parameters: Iterable[nn.Parameter] | Iterable[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[str]]:
    values = list(parameters)
    if not values:
        raise ValueError("FusedAdafactor requires at least one parameter.")
    named = isinstance(values[0], tuple)
    if named:
        if any(not isinstance(value, tuple) or len(value) != 2 for value in values):
            raise TypeError("Do not mix named and unnamed parameters.")
        names = [str(value[0]) for value in values]
        tensors = [value[1] for value in values]
    else:
        names = [f"param_{index}" for index in range(len(values))]
        tensors = values
    if any(not isinstance(parameter, nn.Parameter) for parameter in tensors):
        raise TypeError("FusedAdafactor parameters must be torch.nn.Parameter objects.")
    if len(set(names)) != len(names):
        raise ValueError("FusedAdafactor parameter names must be unique.")
    if len({id(parameter) for parameter in tensors}) != len(tensors):
        raise ValueError("FusedAdafactor does not support duplicate parameters.")
    return tensors, names


class FusedAdafactor(torch.optim.Optimizer):
    """Manual-LR Adafactor that updates and releases one gradient at a time.

    Pass ``model.named_parameters()`` when checkpoint name validation matters.
    Matrix/tensor parameters use FP32 factored second moments over their last
    two dimensions; scalars and vectors use an unfactored FP32 moment.  There
    is no first moment or relative/parameter-scaled learning rate.

    ``install_hooks`` registers post-accumulate hooks. In the default ``device``
    accumulation mode, hooks preserve gradients while ``set_step_enabled(False)``.
    The optional ``cpu`` mode instead copies each early gradient into an FP32
    host buffer and releases ``parameter.grad`` immediately. Call
    ``begin_step`` before a logical optimizer step and ``finish_backward`` (or
    ``step``) after its final backward.  A global norm clip is intentionally
    unsupported because gradients are released during backward.

    Nonfinite checks accumulate into one device flag and synchronize once in
    ``finish_backward``. Detection can therefore follow partial fused updates;
    the optimizer latches an error, rejects saving/further steps, and requires
    model plus optimizer checkpoint reload. ``strict_nonfinite_diagnostics``
    trades per-parameter synchronization for an earlier parameter name.

    The implementation targets one CUDA process. CPU tensors remain useful
    for deterministic reference tests.  In mixed CPU/CUDA models, optimizer
    moments are restored to the inferred CUDA device even while a streamed
    parameter is parked on CPU.
    """

    supports_global_grad_clip = False

    def __init__(
        self,
        parameters: Iterable[nn.Parameter] | Iterable[tuple[str, nn.Parameter]],
        lr: float,
        *,
        eps: float = 1e-30,
        decay_rate: float = -0.8,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        state_device: torch.device | str | None = None,
        strict_nonfinite_diagnostics: bool = False,
        gradient_accumulation_device: str = "device",
        parameter_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        hyperparameters = (lr, eps, decay_rate, clip_threshold, weight_decay)
        if not all(math.isfinite(value) for value in hyperparameters):
            raise ValueError("FusedAdafactor hyperparameters must be finite.")
        if lr < 0 or eps <= 0 or clip_threshold <= 0 or weight_decay < 0:
            raise ValueError("lr/weight_decay must be nonnegative and eps/clip_threshold positive.")
        tensors, names = _split_parameters(parameters)
        unsupported = [parameter.dtype for parameter in tensors if parameter.dtype not in {torch.float32, torch.bfloat16}]
        if unsupported:
            raise TypeError("FusedAdafactor supports only FP32 and BF16 parameters.")
        defaults = {
            "lr": float(lr),
            "eps": float(eps),
            "decay_rate": float(decay_rate),
            "clip_threshold": float(clip_threshold),
            "weight_decay": float(weight_decay),
        }
        super().__init__(tensors, defaults)
        cuda_devices = {parameter.device for parameter in tensors if parameter.device.type == "cuda"}
        if len(cuda_devices) > 1:
            raise ValueError("FusedAdafactor supports one CUDA device per process.")
        if state_device is None:
            self._state_device = next(iter(cuda_devices), torch.device("cpu"))
        else:
            self._state_device = torch.device(state_device)
            if (
                self._state_device.type == "cuda"
                and self._state_device.index is None
                and cuda_devices
            ):
                self._state_device = next(iter(cuda_devices))
        self._parameter_names = tuple(names)
        self._parameter_shapes = tuple(tuple(parameter.shape) for parameter in tensors)
        self._parameter_dtypes = tuple(str(parameter.dtype) for parameter in tensors)
        unique_dtypes = set(self._parameter_dtypes)
        self._master_dtype = next(iter(unique_dtypes)) if len(unique_dtypes) == 1 else "mixed"
        self._name_by_id = dict(zip(map(id, tensors), names))
        known_names = set(names)
        unknown_options = set(parameter_options or ()) - known_names
        if unknown_options:
            raise ValueError(
                "FusedAdafactor parameter_options contains unknown names: "
                f"{sorted(unknown_options)[:3]}."
            )
        self._parameter_options: dict[int, dict[str, Any]] = {}
        for name, parameter in zip(names, tensors):
            options = dict((parameter_options or {}).get(name, {}))
            unexpected = set(options) - {"weight_decay", "decay_to_one"}
            if unexpected:
                raise ValueError(
                    f"Unsupported FusedAdafactor options for {name}: {sorted(unexpected)}."
                )
            decay = float(options.get("weight_decay", weight_decay))
            if not math.isfinite(decay) or decay < 0:
                raise ValueError(f"weight_decay for {name} must be finite and nonnegative.")
            decay_to_one = options.get("decay_to_one", False)
            if not isinstance(decay_to_one, bool):
                raise TypeError(f"decay_to_one for {name} must be bool.")
            stored_options: dict[str, Any] = {"decay_to_one": decay_to_one}
            if "weight_decay" in options:
                stored_options["weight_decay"] = decay
            self._parameter_options[id(parameter)] = stored_options
        self._global_step = 0
        self._pending_step: int | None = None
        self._step_enabled = True
        self._grad_norm_sq: torch.Tensor | None = None
        self._last_gradient_norm = 0.0
        self._found_nonfinite: str | None = None
        self._bad_flag: torch.Tensor | None = None
        self._requires_reload = False
        self._strict_nonfinite_diagnostics = bool(strict_nonfinite_diagnostics)
        self._updated_ids: set[int] = set()
        self._seen_gradient_ids: set[int] = set()
        self._host_gradients: dict[int, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._streamers: list[Any] = []
        self._closed = False
        self._gradient_accumulation_device = "device"
        self.set_gradient_accumulation_device(gradient_accumulation_device)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def gradient_norm(self) -> float:
        """L2 norm of gradients consumed by the most recently finished step."""

        return self._last_gradient_norm

    @property
    def updated_parameter_count(self) -> int:
        return len(self._updated_ids)

    @property
    def step_enabled(self) -> bool:
        return self._step_enabled

    @property
    def requires_reload(self) -> bool:
        return self._requires_reload

    @property
    def gradient_accumulation_device(self) -> str:
        return self._gradient_accumulation_device

    def set_gradient_accumulation_device(self, device: str) -> None:
        """Select legacy device accumulation or FP32 CPU gradient accumulation."""

        if device not in _GRADIENT_ACCUMULATION_DEVICES:
            raise ValueError("gradient_accumulation_device must be 'device', 'cuda', or 'cpu'.")
        normalized = "device" if device == "cuda" else device
        if self._pending_step is not None or self._host_gradients:
            raise RuntimeError("Cannot change gradient accumulation device during an active step.")
        self._gradient_accumulation_device = normalized
        for streamer in self._streamers:
            configure = getattr(streamer, "set_require_all_updates", None)
            if configure is not None:
                configure(normalized != "cpu")

    def set_step_enabled(self, enabled: bool) -> None:
        """Enable fused updates for the current accumulation boundary."""

        self._step_enabled = bool(enabled)
        for streamer in self._streamers:
            streamer.set_step_enabled(self._step_enabled)

    def set_last_group_step_boundary(self, enabled: bool) -> None:
        """Alias used by sequential shape-group training loops."""

        self.set_step_enabled(enabled)

    def reset_norm(self) -> None:
        self._grad_norm_sq = None
        self._last_gradient_norm = 0.0

    def begin_step(self) -> None:
        """Start one logical optimizer step without changing any parameter."""

        if self._closed:
            raise RuntimeError("FusedAdafactor is closed.")
        if self._requires_reload:
            raise RuntimeError("FusedAdafactor is invalid after a failed step; reload a checkpoint.")
        if self._pending_step is not None:
            raise RuntimeError("FusedAdafactor step is already active.")
        self._pending_step = self._global_step + 1
        self._found_nonfinite = None
        self._bad_flag = None
        self._updated_ids.clear()
        self._seen_gradient_ids.clear()
        self._host_gradients.clear()
        self.reset_norm()

    def _iter_parameters(self) -> Iterator[tuple[nn.Parameter, dict[str, Any]]]:
        for group in self.param_groups:
            for parameter in group["params"]:
                yield parameter, group

    def _resolve_state_device(self, reference: torch.device) -> torch.device:
        device = self._state_device
        if device.type == "cuda" and device.index is None and reference.type == "cuda":
            device = reference
            self._state_device = device
        return device

    def _state_for(self, parameter: nn.Parameter, gradient: torch.Tensor) -> dict[str, Any]:
        state = self.state[parameter]
        device = self._resolve_state_device(gradient.device)
        if device.type == "cpu" and gradient.device.type == "cuda":
            device = gradient.device
            self._state_device = device
        if gradient.device != device:
            raise RuntimeError(
                f"FusedAdafactor gradient for {self._name_by_id[id(parameter)]} is on "
                f"{gradient.device}, expected {device}."
            )
        if not state:
            if gradient.ndim >= 2:
                state["exp_avg_sq_row"] = torch.zeros(
                    gradient.shape[:-1], device=device, dtype=torch.float32
                )
                state["exp_avg_sq_col"] = torch.zeros(
                    gradient.shape[:-2] + gradient.shape[-1:], device=device, dtype=torch.float32
                )
            else:
                state["exp_avg_sq"] = torch.zeros_like(gradient, device=device, dtype=torch.float32)
            state["step"] = 0
        for key in _MOMENT_KEYS:
            value = state.get(key)
            if isinstance(value, torch.Tensor) and (value.device != device or value.dtype != torch.float32):
                state[key] = value.to(device=device, dtype=torch.float32)
        return state

    def _accumulate_norm(self, gradient: torch.Tensor) -> None:
        square = gradient.float().square().sum().detach()
        if self._grad_norm_sq is None:
            self._grad_norm_sq = square
        else:
            self._grad_norm_sq.add_(square)

    def _accumulate_bad_flag(self, bad: torch.Tensor, parameter: nn.Parameter) -> bool:
        bad = bad.detach()
        if self._bad_flag is None:
            self._bad_flag = bad
        else:
            self._bad_flag.logical_or_(bad)
        if self._strict_nonfinite_diagnostics and bool(bad.item()):
            self._found_nonfinite = self._name_by_id[id(parameter)]
            return True
        return False

    def _mark_streamers(self, parameter: nn.Parameter, *, failed: bool = False) -> None:
        for streamer in self._streamers:
            if failed:
                streamer.mark_failed(parameter)
            else:
                streamer.mark_updated(parameter)

    @staticmethod
    def _validate_gradient(gradient: torch.Tensor) -> None:
        if gradient.is_sparse:
            raise RuntimeError("FusedAdafactor does not support sparse gradients.")

    @torch.no_grad()
    def _stash_gradient_on_host(self, parameter: nn.Parameter) -> bool:
        gradient = parameter.grad
        if gradient is None:
            return False
        self._validate_gradient(gradient)
        if self._pending_step is None:
            self.begin_step()
        parameter_id = id(parameter)
        self._seen_gradient_ids.add(parameter_id)
        if gradient.device.type == "cuda":
            host_gradient = torch.empty(
                gradient.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            # A blocking copy deliberately keeps the autograd and trainable-block
            # streamer lifetimes simple: the hook does not return until CUDA is
            # finished reading the gradient storage that is released below.
            host_gradient.copy_(gradient.detach(), non_blocking=False)
        else:
            host_gradient = gradient.detach().to(device="cpu", dtype=torch.float32).clone()
        accumulated = self._host_gradients.get(parameter_id)
        if accumulated is None:
            self._host_gradients[parameter_id] = host_gradient
        else:
            accumulated.add_(host_gradient)
        parameter.grad = None
        return True

    def _take_accumulated_gradient(self, parameter: nn.Parameter) -> torch.Tensor | None:
        parameter_id = id(parameter)
        current = parameter.grad
        if current is not None:
            self._validate_gradient(current)
            self._seen_gradient_ids.add(parameter_id)
        previous = self._host_gradients.pop(parameter_id, None)
        if previous is None:
            return current
        if current is None:
            return previous
        current_fp32 = current.detach().float()
        return current_fp32.add(previous.to(device=current.device, non_blocking=False))

    @torch.no_grad()
    def _apply_gradient(
        self,
        parameter: nn.Parameter,
        gradient: torch.Tensor,
        group: dict[str, Any],
    ) -> bool:
        if self._requires_reload:
            raise RuntimeError("FusedAdafactor is invalid after a failed step; reload a checkpoint.")
        if self._found_nonfinite is not None:
            self._mark_streamers(parameter, failed=True)
            return False
        parameter_id = id(parameter)
        if parameter_id in self._updated_ids:
            raise RuntimeError(f"Parameter {self._name_by_id[parameter_id]} was updated twice in one step.")

        device = self._resolve_state_device(gradient.device)
        if device.type == "cpu" and gradient.device.type == "cuda":
            device = gradient.device
            self._state_device = device
        grad = gradient.detach().to(device=device, dtype=torch.float32)
        parameter_work = parameter if parameter.device == device else parameter.detach().to(device=device)
        pre_bad = torch.logical_or(
            torch.logical_not(torch.isfinite(grad).all()),
            torch.logical_not(torch.isfinite(parameter_work).all()),
        )
        if self._accumulate_bad_flag(pre_bad, parameter):
            self._mark_streamers(parameter, failed=True)
            return False
        state = self._state_for(parameter, grad)
        for key in _MOMENT_KEYS:
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                state_bad = torch.logical_not(torch.isfinite(value).all())
                if self._accumulate_bad_flag(state_bad, parameter):
                    self._mark_streamers(parameter, failed=True)
                    return False

        step = int(self._pending_step)
        beta2 = 1.0 - math.pow(step, float(group["decay_rate"]))
        squared = grad.square().add_(float(group["eps"]))
        if grad.ndim >= 2:
            row = state["exp_avg_sq_row"]
            column = state["exp_avg_sq_col"]
            row.mul_(beta2).add_(squared.mean(dim=-1), alpha=1.0 - beta2)
            column.mul_(beta2).add_(squared.mean(dim=-2), alpha=1.0 - beta2)
            row_factor = (
                row / row.mean(dim=-1, keepdim=True).clamp_min_(float(group["eps"]))
            ).rsqrt_().unsqueeze(-1)
            column_factor = column.rsqrt().unsqueeze(-2)
            update = grad.mul(row_factor).mul_(column_factor)
        else:
            moment = state["exp_avg_sq"]
            moment.mul_(beta2).add_(squared, alpha=1.0 - beta2)
            update = moment.rsqrt().mul_(grad)
        update_rms = update.square().mean().sqrt()
        update.div_(torch.clamp(update_rms / float(group["clip_threshold"]), min=1.0))
        update_bad = torch.logical_not(torch.isfinite(update).all())
        if self._accumulate_bad_flag(update_bad, parameter):
            self._mark_streamers(parameter, failed=True)
            return False

        parameter_fp32 = parameter_work if parameter_work.dtype == torch.float32 else parameter_work.float()
        learning_rate = float(group["lr"])
        options = self._parameter_options[parameter_id]
        weight_decay = float(options.get("weight_decay", group["weight_decay"]))
        if weight_decay:
            if options["decay_to_one"]:
                parameter_fp32.add_(parameter_fp32 - 1.0, alpha=-learning_rate * weight_decay)
            else:
                parameter_fp32.mul_(1.0 - learning_rate * weight_decay)
        parameter_fp32.add_(update, alpha=-learning_rate)
        if parameter_work.dtype == torch.bfloat16:
            stochastic_copy_(parameter_work, parameter_fp32)
        if parameter_work is not parameter:
            parameter.copy_(parameter_work.to(device=parameter.device, dtype=parameter.dtype))
        result_bad = torch.logical_not(torch.isfinite(parameter_work).all())
        if self._accumulate_bad_flag(result_bad, parameter):
            self._mark_streamers(parameter, failed=True)
            return False
        self._accumulate_norm(grad)
        state["step"] = step
        self._updated_ids.add(parameter_id)
        self._mark_streamers(parameter)
        parameter.grad = None
        return True

    @torch.no_grad()
    def step_parameter(self, parameter: nn.Parameter, group: dict[str, Any] | None = None) -> bool:
        """Apply one fused update, returning whether a gradient was consumed."""

        gradient = self._take_accumulated_gradient(parameter)
        if gradient is None:
            return False
        if self._pending_step is None:
            self.begin_step()
        if not self._step_enabled:
            if self._gradient_accumulation_device == "cpu":
                parameter.grad = gradient
                return self._stash_gradient_on_host(parameter)
            return False
        group = group or next(group for candidate, group in self._iter_parameters() if candidate is parameter)
        return self._apply_gradient(parameter, gradient, group)

    # Compatibility with fused optimizer patch call sites.
    step_param = step_parameter

    def install_hooks(self, streamer: Any | None = None) -> "FusedAdafactor":
        """Install one post-accumulate update hook per trainable parameter."""

        if self._closed:
            raise RuntimeError("FusedAdafactor is closed.")
        if streamer is not None and streamer not in self._streamers:
            self._streamers.append(streamer)
            configure = getattr(streamer, "set_require_all_updates", None)
            if configure is not None:
                configure(self._gradient_accumulation_device != "cpu")
            streamer.set_step_enabled(self._step_enabled)
        if self._hook_handles:
            return self

        group_by_id = {
            id(parameter): group for parameter, group in self._iter_parameters() if parameter.requires_grad
        }
        for parameter, _ in self._iter_parameters():
            if not parameter.requires_grad:
                continue

            def update_after_accumulate(
                tensor: torch.Tensor,
                *,
                parameter: nn.Parameter = parameter,
                group: dict[str, Any] = group_by_id[id(parameter)],
            ) -> None:
                if tensor is not parameter:
                    raise RuntimeError("FusedAdafactor hook received an unexpected tensor.")
                if self._step_enabled:
                    self.step_parameter(parameter, group)
                elif self._gradient_accumulation_device == "cpu":
                    self._stash_gradient_on_host(parameter)

            self._hook_handles.append(parameter.register_post_accumulate_grad_hook(update_after_accumulate))
        return self

    def finish_backward(self) -> float:
        """Finish the active logical step and return its pre-update grad norm."""

        if self._pending_step is None:
            return self._last_gradient_norm
        if not self._step_enabled:
            return self._last_gradient_norm
        if self._gradient_accumulation_device == "cpu":
            for parameter, group in self._iter_parameters():
                if id(parameter) in self._host_gradients:
                    self.step_parameter(parameter, group)
        bad = self._bad_flag is not None and bool(self._bad_flag.item())
        if bad or self._found_nonfinite is not None:
            name = self._found_nonfinite or "an asynchronously checked parameter"
            self._pending_step = None
            self._requires_reload = True
            raise FloatingPointError(
                "Nonfinite gradient, parameter, or Adafactor state for "
                f"{name}; this fused step may be partially applied and requires checkpoint reload."
            )
        if self._gradient_accumulation_device == "cpu":
            expected = self._seen_gradient_ids
        else:
            expected = {
                id(parameter) for parameter, _ in self._iter_parameters() if parameter.requires_grad
            }
        missing = expected - self._updated_ids
        if missing:
            names = sorted(self._name_by_id[parameter_id] for parameter_id in missing)
            self._pending_step = None
            self._requires_reload = True
            raise RuntimeError(
                "FusedAdafactor did not receive/update every trainable parameter; "
                f"missing {names[:3]}. The partial step requires checkpoint reload."
            )
        self._global_step = self._pending_step
        self._pending_step = None
        self._last_gradient_norm = (
            float(self._grad_norm_sq.sqrt().item()) if self._grad_norm_sq is not None else 0.0
        )
        return self._last_gradient_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Finish hook updates, or perform a conventional per-parameter step."""

        loss = closure() if closure is not None else None
        if not self._step_enabled:
            return loss
        if self._pending_step is None and any(parameter.grad is not None for parameter, _ in self._iter_parameters()):
            self.begin_step()
        if self._pending_step is not None:
            for parameter, group in self._iter_parameters():
                if parameter.grad is not None:
                    self.step_parameter(parameter, group)
            self.finish_backward()
        return loss

    def clip_grad_norm_(self, *_args, **_kwargs) -> None:
        raise RuntimeError("FusedAdafactor cannot apply global gradient clipping after fused updates.")

    def state_dict(self) -> dict[str, Any]:
        if self._requires_reload:
            raise RuntimeError("Cannot save FusedAdafactor after a failed partial step; reload first.")
        if self._pending_step is not None or self._host_gradients:
            raise RuntimeError("Cannot save FusedAdafactor with an uncommitted optimizer step.")
        result = super().state_dict()
        result["fused_adafactor"] = {
            "version": 1,
            "optimizer_kind": _OPTIMIZER_KIND,
            "master_dtype": self._master_dtype,
            "parameter_names": self._parameter_names,
            "parameter_shapes": self._parameter_shapes,
            "parameter_dtypes": self._parameter_dtypes,
            "global_step": self._global_step,
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Keep CUDA moment tensors single-copy during resume. Optimizer.load_state_dict
        # rebuilds its mappings; only the top-level metadata entry is removed here.
        incoming = dict(state_dict)
        metadata = incoming.pop("fused_adafactor", None)
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ValueError("FusedAdafactor checkpoint metadata is missing or unsupported.")
        if metadata.get("optimizer_kind") != _OPTIMIZER_KIND:
            raise ValueError("FusedAdafactor checkpoint optimizer kind does not match.")
        if metadata.get("master_dtype") != self._master_dtype:
            raise ValueError("FusedAdafactor checkpoint master dtype does not match.")
        saved_names = tuple(metadata.get("parameter_names", ()))
        saved_shapes = tuple(tuple(shape) for shape in metadata.get("parameter_shapes", ()))
        saved_dtypes = tuple(metadata.get("parameter_dtypes", ()))
        if saved_names != self._parameter_names:
            raise ValueError("FusedAdafactor checkpoint parameter names/order do not match.")
        if saved_shapes != self._parameter_shapes:
            raise ValueError("FusedAdafactor checkpoint parameter shapes do not match.")
        if saved_dtypes != self._parameter_dtypes:
            raise ValueError("FusedAdafactor checkpoint parameter dtypes do not match.")
        super().load_state_dict(incoming)
        # torch.optim normally casts floating state to each parameter's dtype.
        # Restore moments from the original checkpoint tensors so a BF16 CPU
        # master cannot quantize FP32 state during the temporary default load.
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
            for parameter, saved_id in zip(group["params"], saved_group["params"]):
                current_state = self.state[parameter]
                saved_state = state_dict["state"].get(saved_id, {})
                for key in _MOMENT_KEYS:
                    value = saved_state.get(key)
                    if isinstance(value, torch.Tensor):
                        current_state[key] = value.detach().to(
                            device=self._state_device, dtype=torch.float32
                        ).clone()
                if saved_state:
                    current_state["step"] = int(saved_state.get("step", 0))
                else:
                    current_state.clear()
        self._global_step = int(metadata.get("global_step", 0))
        self._pending_step = None
        self._found_nonfinite = None
        self._bad_flag = None
        self._requires_reload = False
        self._updated_ids.clear()
        self._seen_gradient_ids.clear()
        self._host_gradients.clear()
        self.reset_norm()

    def close(self) -> None:
        """Remove hooks. Optimizer state remains available for checkpointing."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._host_gradients.clear()
        self._closed = True

