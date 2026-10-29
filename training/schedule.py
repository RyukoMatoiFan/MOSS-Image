"""Inference schedule used by the released MOSS-Image pipeline.

Sigmas are a shifted linspace, then snapped to the nearest point on the training
grid (shift 1, 1000 steps) so sampling stays on times the model was trained on.
The terminal sigma is 0, appended by the Diffusers Euler scheduler.
"""

from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config


def shifted_grid(count: int, shift: float, total: int = 1000) -> torch.Tensor:
    """Build the descending shifted sigma grid from the released scheduler.

    Args:
        count: Number of sigmas.
        shift: Schedule shift. Inference uses 5. The training grid uses 1.
        total: Training timestep count, used only to place the lower end.

    Returns:
        ``[count]`` float32 sigmas, descending, in ``(0, 1]``.
    """

    if count < 1 or shift <= 0 or total < 1:
        raise ValueError("count, shift, and total must be positive.")
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def snap_sigmas(requested: torch.Tensor, training_shift: float = 1.0, total: int = 1000) -> torch.Tensor:
    """Snap descending sigmas onto the ascending training grid.

    Args:
        requested: Candidate sigmas, typically the inference shift grid.
        training_shift: Shift of the grid the model was trained on.
        total: Training grid length.

    Returns:
        Unique descending sigmas, each equal to a training-grid value.
    """

    training = shifted_grid(total, training_shift, total).flip(0)
    upper = torch.searchsorted(training, requested).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (requested - training[lower]).abs() <= (training[upper] - requested).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler sampler with the released V2 nearest-grid schedule."""

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps: int = 50, device: torch.device | str | None = None) -> None:
        """Install snapped sigmas.

        Args:
            num_inference_steps: Requested step count. Snapping can only shorten it.
            device: Device of the scheduler tensors.
        """

        if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
            raise ValueError("Steps and shifts must be positive.")
        sigmas = snap_sigmas(
            shifted_grid(num_inference_steps, self.config.shift, self.config.num_train_timesteps),
            self.config.training_shift,
            self.config.num_train_timesteps,
        )
        # The sigmas are already shifted. Keep the parent from shifting them again.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())
