"""Memory-conscious optimizer construction using public PyTorch APIs."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def parameter_groups(named_parameters, weight_decay: float, policy: str = "all", *,
                     qk_norm_decay_to_one: bool = False, depth: int = 0) -> list[dict]:
    """Group decay by tensor role while preserving stable parameter order."""
    buckets: dict[tuple[float, bool], dict] = {}
    embedding_prefixes = ("video_embedder.", "text_embedder.proj.", "time_embedding.")
    last_norms = {f"layers.{depth - 1}.{name}.weight" for name in ("q_norm", "k_norm")}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        exclude = parameter.ndim < 2 or "norm" in name or name.endswith(".bias") or name.startswith(embedding_prefixes)
        decay = 0.0 if policy == "matrix" and exclude else weight_decay
        to_one = qk_norm_decay_to_one and name in last_norms
        key = (weight_decay if to_one else decay, to_one)
        group = buckets.setdefault(key, {"params": [], "names": [], "weight_decay": 0.0 if to_one else decay,
                                         "decay_to_one": to_one, "decay_to_one_coef": weight_decay if to_one else 0.0})
        group["params"].append(parameter)
        group["names"].append(name)
    return list(buckets.values())


def install_decay_to_one(optimizer: torch.optim.Optimizer) -> None:
    """Apply explicitly selected decay toward one before the ordinary update."""
    @torch.no_grad()
    def decay(optimizer, args, kwargs):
        for group in optimizer.param_groups:
            coefficient = group.get("decay_to_one_coef", 0.0)
            if group.get("decay_to_one") and coefficient:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.add_(parameter - 1, alpha=-group["lr"] * coefficient)
    optimizer.register_step_pre_hook(decay)


def build_adafactor(
    parameters: Iterable[torch.Tensor], lr: float, weight_decay: float,
) -> torch.optim.Optimizer:
    """Build native Adafactor for resident parameters and CPU block masters.

    ``lr`` is passed unchanged to PyTorch: its relative step is
    ``min(lr, 1 / sqrt(step))`` and its absolute update scale multiplies that
    by ``max(0.001, RMS(parameter))``. Decoupled weight decay uses ``lr``
    directly. These are PyTorch Adafactor semantics, not an AdamW-equivalent
    absolute learning rate; AdamW learning rates must be tuned for this optimizer.

    Matrices keep factored row/column second moments without a first moment.
    Vectors keep a full second moment. Single-tensor updates avoid foreach
    intermediates spanning the parameter list and support mixed CPU/CUDA masters.
    Parameters and gradients still occupy full storage; matrix-sized temporary
    updates are allocated during each step. This does not itself offload anything.
    """
    optimizer = getattr(torch.optim, "Adafactor", None)
    if optimizer is None:
        raise RuntimeError("Adafactor requires a PyTorch release exposing torch.optim.Adafactor.")
    return optimizer(parameters, lr=lr, weight_decay=weight_decay, foreach=False)
