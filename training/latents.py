"""FLUX.2 latent patching and image sizing for MOSS-Image.

The VAE emits 32 channels at stride 8. A 2x2 pack produces the 128-channel tokens
the DiT embeds, at an effective stride of 16. Batch-norm statistics stored on the
VAE are applied after packing and inverted before decoding, matching the released
pipeline.
"""

from __future__ import annotations

import math

import torch


def fit_dimensions(width: int, height: int, max_pixels: int, multiple: int = 16) -> tuple[int, int]:
    """Resize an image's dimensions to a stride-aligned size within a pixel budget.

    Aspect ratio is preserved up to rounding onto ``multiple``. Images smaller than
    the budget are scaled up. The result is at least ``multiple`` on both sides.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        max_pixels: Maximum ``width * height`` after rounding.
        multiple: Alignment. MOSS-Image uses 16, the VAE stride times the 2x2 pack.

    Returns:
        ``(width, height)`` multiples of ``multiple`` whose product is at most
        ``max_pixels`` when that is possible.
    """

    if width < 1 or height < 1 or multiple < 1:
        raise ValueError("Dimensions and alignment must be positive.")
    if max_pixels < multiple * multiple:
        raise ValueError("max_pixels is smaller than one aligned block.")
    scale = math.sqrt(max_pixels / (width * height))
    resized_w = max(multiple, int(width * scale) // multiple * multiple)
    resized_h = max(multiple, int(height * scale) // multiple * multiple)
    while resized_w * resized_h > max_pixels and (resized_w > multiple or resized_h > multiple):
        if resized_w >= resized_h and resized_w > multiple:
            resized_w -= multiple
        else:
            resized_h -= multiple
    return resized_w, resized_h


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack each 2x2 neighborhood of a 32-channel FLUX.2 latent into 128 channels.

    Args:
        latents: ``[batch, 32, height, width]`` with even spatial size.

    Returns:
        ``[batch, 128, height // 2, width // 2]``.
    """

    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError("Latent spatial size must be even before 2x2 packing.")
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 1, 3, 5, 2, 4)
    return packed.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Invert ``patchify_latents``.

    Args:
        latents: ``[batch, 128, height, width]``.

    Returns:
        ``[batch, 32, height * 2, width * 2]``.
    """

    batch, channels, height, width = latents.shape
    if channels % 4:
        raise ValueError("Packed latents must have a channel count divisible by 4.")
    unpacked = latents.reshape(batch, channels // 4, 2, 2, height, width)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)
    return unpacked.reshape(batch, channels // 4, height * 2, width * 2)


def normalize_patched(latents: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply the FLUX.2 batch-norm whitening used before the DiT.

    Args:
        latents: Packed latents ``[batch, 128, height, width]``.
        mean: ``[128]`` running mean.
        variance: ``[128]`` running variance.
        eps: Batch-norm epsilon from the VAE config.

    Returns:
        Whitened latents, same shape as ``latents``.
    """

    std = torch.sqrt(variance.to(latents) + eps).view(1, -1, 1, 1)
    return (latents - mean.to(latents).view(1, -1, 1, 1)) / std


def denormalize_patched(latents: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor, eps: float) -> torch.Tensor:
    """Invert ``normalize_patched`` before the VAE decoder.

    Args:
        latents: DiT latents ``[batch, 128, height, width]``.
        mean: ``[128]`` running mean.
        variance: ``[128]`` running variance.
        eps: Batch-norm epsilon.

    Returns:
        Latents in the VAE's native scale.
    """

    std = torch.sqrt(variance.to(latents) + eps).view(1, -1, 1, 1)
    return latents * std + mean.to(latents).view(1, -1, 1, 1)
