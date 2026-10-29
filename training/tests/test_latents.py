"""Latent packing and the inference schedule."""

from __future__ import annotations

import torch

from training.latents import fit_dimensions, patchify_latents, unpatchify_latents
from training.schedule import MOSSImageScheduler, shifted_grid


def test_patchify_round_trip() -> None:
    """2x2 packing is invertible and multiplies the channel count by 4."""

    latents = torch.arange(2 * 32 * 8 * 6, dtype=torch.float32).reshape(2, 32, 8, 6)
    packed = patchify_latents(latents)
    assert packed.shape == (2, 128, 4, 3)
    assert torch.equal(unpatchify_latents(packed), latents)


def test_fit_dimensions_respects_budget_and_stride() -> None:
    """A wide image stays wide, lands on multiples of 16, and fits the pixel budget."""

    width, height = fit_dimensions(2000, 1000, 1024 * 1024)
    assert width % 16 == 0 and height % 16 == 0
    assert width * height <= 1024 * 1024
    assert width > height
    assert abs(width / height - 2) < 0.05


def test_sampler_snaps_onto_the_training_grid() -> None:
    """Shift 5 sigmas are values from the shift-1 training grid, descending toward 0."""

    scheduler = MOSSImageScheduler(shift=5.0, training_shift=1.0)
    scheduler.set_timesteps(8)
    training = shifted_grid(1000, 1.0).flip(0)
    body = scheduler.sigmas[:-1]
    assert scheduler.sigmas[-1].item() == 0
    assert torch.all(body[:-1] > body[1:])
    for sigma in body:
        assert torch.any(torch.isclose(training, sigma, atol=1e-6))
