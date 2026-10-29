"""Repeatable held-out velocity losses using a separate CPU random stream."""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def evaluate_fixed(
    model: torch.nn.Module,
    prepared: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    sigmas: tuple[float, ...],
    seed: int,
    total_timesteps: int,
    device: torch.device,
    mixed_precision: str,
    score: Callable,
) -> dict[str, float]:
    """Evaluate each row separately, keeping training RNG and mode unchanged."""
    was_training = model.training
    manager = getattr(model, "block_swap_manager", None)
    if manager is not None and hasattr(manager, "commit"):
        manager.commit()
    model.eval()
    try:
        values = []
        result = {}
        for sigma in sigmas:
            generator = torch.Generator(device="cpu").manual_seed(seed)

            class FixedMatcher:
                def sample_sigma(self, batch, target_device, generator=None):
                    return torch.full((batch,), sigma, dtype=torch.float32, device=target_device)

                def timestep(self, drawn):
                    return drawn * total_timesteps

                def noisy_latent(self, clean, drawn, unused_noise):
                    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean.device)
                    scale = drawn.view(-1, *([1] * (clean.ndim - 1)))
                    return clean * (1 - scale) + noise * scale, noise - clean

            # score's ordinary noise draw is isolated too, so evaluation does not
            # change the training stream even though the matcher replaces it.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=mixed_precision == "bf16",
            ):
                error = torch.zeros((), device=device)
                count = 0
                for example in prepared:
                    row_error, row_count = score(model, FixedMatcher(), [example], device)
                    error += row_error
                    count += row_count
                value = float(error / count)
            result[f"eval_loss@{sigma:g}"] = value
            values.append(value)
        result["eval_loss"] = sum(values) / len(values)
        return result
    finally:
        model.train(was_training)
