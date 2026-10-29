"""LoRA adapters stored in the ComfyUI diffusion-model key layout.

Keys look like ``diffusion_model.layers.0.proj.q.lora_A.weight``. The frozen base
linear stays in place. ``A`` is ``[rank, in]``, ``B`` is ``[out, rank]``, and the
forward adds ``(alpha / rank) * B @ A``. ``B`` starts at zero, so a fresh adapter
matches the base model.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear plus a low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        # ``None`` keeps FP32 adapter math outside autocast. Performance setup may
        # request an explicit compute dtype without changing the FP32 master
        # parameters stored in the state dict and optimizer.
        self.lora_compute_dtype: torch.dtype | None = None
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def compute_dtype(self) -> torch.dtype:
        if hasattr(self.base, "compute_dtype"):
            return self.base.compute_dtype
        return self.base.weight.dtype

    @property
    def weight(self) -> torch.Tensor:
        """Frozen base weight, so callers can read its dtype and shape."""

        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Frozen base bias, or ``None`` when the linear has none."""

        return self.base.bias

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the frozen linear and the low-rank update.

        Args:
            tokens: ``[..., in_features]``.

        Returns:
            ``[..., out_features]`` in the base result's dtype.
        """

        result = self.base(tokens)
        compute_dtype = self.lora_compute_dtype
        if compute_dtype is None and torch.is_autocast_enabled(tokens.device.type):
            compute_dtype = torch.get_autocast_dtype(tokens.device.type)
        if compute_dtype is None:
            # Preserve the original FP32 path when autocast is disabled.
            adapter_input = tokens.to(self.lora_A.dtype)
            adapter_a = self.lora_A
            adapter_b = self.lora_B
        else:
            # Casting the small trainable matrices avoids upcasting the much
            # larger token tensor to the FP32 master-parameter dtype.
            adapter_input = tokens.to(compute_dtype)
            adapter_a = self.lora_A.to(compute_dtype)
            adapter_b = self.lora_B.to(compute_dtype)
        update = F.linear(F.linear(adapter_input, adapter_a), adapter_b)
        return result + update.to(result.dtype) * self.scale


def _parent_and_name(root: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    """Return the parent module and the final attribute of a dotted name.

    Args:
        root: Module that contains ``qualified``.
        qualified: Dotted path such as ``layers.0.proj.q``.

    Returns:
        Parent module and the child name. Numeric names index a ``ModuleList``.
    """

    parent: nn.Module = root
    parts = qualified.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _replace(root: nn.Module, qualified: str, module: nn.Module) -> None:
    """Replace a submodule addressed by a dotted name."""

    parent, name = _parent_and_name(root, qualified)
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def target_linear_names(model: nn.Module, targets: str) -> list[str]:
    """List linear modules that receive adapters.

    Args:
        model: A ``MOSSImageTransformer``.
        targets: ``joint`` adapts joint blocks only. ``joint_and_text`` adapts
            joint blocks and the text adapter. The legacy ``blocks`` name keeps
            the same behavior as ``joint_and_text``. ``blocks_and_io`` also
            adapts the video embedder and velocity head.
            Time embedding and modulation stay frozen in both modes.

    Returns:
        Dotted module names, in ``named_modules`` order.
    """

    accepted = {"joint", "joint_and_text", "blocks", "blocks_and_io"}
    if targets not in accepted:
        choices = ", ".join(sorted(accepted))
        raise ValueError(f"lora targets must be one of: {choices}.")
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        in_joint = name.startswith("layers.")
        in_text = name.startswith("text_embedder.")
        in_io = name in {"video_embedder", "final_linear_video"}
        include_text = targets in {"joint_and_text", "blocks", "blocks_and_io"}
        if in_joint or (include_text and in_text) or (targets == "blocks_and_io" and in_io):
            names.append(name)
    if not names:
        raise ValueError("No linear layers matched the LoRA target set.")
    return names


def inject_lora(model: nn.Module, rank: int, alpha: float, targets: str = "blocks") -> list[str]:
    """Freeze the transformer and attach LoRA residuals.

    Args:
        model: Transformer to adapt. Base parameters are frozen.
        rank: Adapter rank.
        alpha: Scale numerator. The forward scale is ``alpha / rank``.
        targets: See ``target_linear_names``.

    Returns:
        Names of the adapted modules.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = target_linear_names(model, targets)
    for name in names:
        parent, child = _parent_and_name(model, name)
        base = parent[int(child)] if child.isdigit() else getattr(parent, child)
        _replace(model, name, LoRALinear(base, rank, alpha))
    object.__setattr__(model, "_moss_lora_targets", targets)
    return names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable adapter parameters.

    Args:
        model: Transformer after ``inject_lora``.

    Returns:
        ``lora_A`` and ``lora_B`` parameters.
    """

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect ComfyUI-layout adapter tensors.

    Args:
        model: Transformer after ``inject_lora``.

    Returns:
        CPU tensors keyed by ``diffusion_model.<path>.lora_A.weight``,
        ``lora_B.weight``, and a scalar ``alpha``.
    """

    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"diffusion_model.{name}"
        state[f"{prefix}.lora_A.weight"] = module.lora_A.detach().float().cpu().contiguous()
        state[f"{prefix}.lora_B.weight"] = module.lora_B.detach().float().cpu().contiguous()
        state[f"{prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    if not state:
        raise ValueError("No LoRA modules are attached.")
    return state


def save_lora(model: nn.Module, path: Path) -> None:
    """Write the adapter to a safetensors file.

    Args:
        model: Transformer after ``inject_lora``.
        path: Destination file. Parent directories are created.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    targets = getattr(model, "_moss_lora_targets", "blocks")
    save_file(lora_state_dict(model), str(path), metadata={"moss_lora_targets": targets})


def lora_spec(path: Path) -> tuple[int, float, str]:
    """Read rank, alpha, and target set from an adapter file.

    Args:
        path: Safetensors file written by ``save_lora``.

    Returns:
        Rank, alpha, and the exact stored target preset when metadata is present.
    """

    state = load_file(str(path))
    rank = None
    alpha = None
    names: list[str] = []
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            rank = int(value.shape[0])
            names.append(key)
        elif key.endswith(".alpha") and alpha is None:
            alpha = float(value)
    if rank is None:
        raise ValueError(f"No LoRA matrices in {path}")
    targets = None
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        targets = (reader.metadata() or {}).get("moss_lora_targets")
    accepted = {"joint", "joint_and_text", "blocks", "blocks_and_io"}
    if targets not in accepted:
        if any("video_embedder" in key or "final_linear_video" in key for key in names):
            targets = "blocks_and_io"
        elif any("text_embedder" in key for key in names):
            targets = "blocks"
        else:
            targets = "joint"
    return rank, float(rank if alpha is None else alpha), targets


def _lora_name(key: str, suffix: str) -> str:
    """Return the module path stored under a ``diffusion_model.`` LoRA key."""

    return key[len("diffusion_model.") : -len(suffix)]


def load_lora(model: nn.Module, path: Path) -> None:
    """Load an adapter written by ``save_lora`` into an already injected model.

    Both matrices are required. A stored ``alpha`` replaces the scale from
    ``inject_lora``, so reloading rank 4 with alpha 8 applies scale 2. Every
    key is checked before any weight or scale changes, so a rejected file
    leaves the injected model untouched.

    Args:
        model: Transformer after ``inject_lora`` with the same rank and targets.
        path: Safetensors file.
    """

    state = load_file(str(path))
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    alphas: dict[str, float] = {}
    pending: list[tuple[nn.Parameter, torch.Tensor]] = []
    for key, value in state.items():
        if not key.startswith("diffusion_model."):
            continue
        if key.endswith(".alpha"):
            name = _lora_name(key, ".alpha")
            if name not in modules:
                raise ValueError(f"LoRA key does not match an injected module: {name}")
            alphas[name] = float(value)
            continue
        if key.endswith(".lora_A.weight"):
            name = _lora_name(key, ".lora_A.weight")
            slot = "lora_A"
            seen_a.add(name)
        elif key.endswith(".lora_B.weight"):
            name = _lora_name(key, ".lora_B.weight")
            slot = "lora_B"
            seen_b.add(name)
        else:
            raise ValueError(f"Unrecognized LoRA key: {key}")
        if name not in modules:
            raise ValueError(f"LoRA key does not match an injected module: {name}")
        parameter = getattr(modules[name], slot)
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} vs {tuple(parameter.shape)}")
        pending.append((parameter, value))
    missing = sorted(name for name in modules if name not in seen_a or name not in seen_b)
    if missing:
        raise ValueError(f"LoRA checkpoint is missing modules: {missing[:4]}")
    for parameter, value in pending:
        parameter.data.copy_(value.to(dtype=parameter.dtype, device=parameter.device))
    for name, alpha in alphas.items():
        module = modules[name]
        module.alpha = alpha
        module.scale = alpha / module.rank
