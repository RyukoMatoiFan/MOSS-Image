"""RGB image loading and the FLUX.2 encode used by precache and sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from training.latents import denormalize_patched, normalize_patched, patchify_latents, unpatchify_latents


def load_rgb(path: Path, width: int, height: int) -> torch.Tensor:
    """Load an image, resize it, and scale pixels to ``[-1, 1]``.

    Args:
        path: Image file.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Float tensor ``[3, height, width]`` on CPU.
    """

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1)


def save_rgb(pixels: torch.Tensor, path: Path) -> None:
    """Write one image in ``[-1, 1]``.

    Args:
        pixels: ``[3, height, width]``.
        path: Destination PNG or JPEG.
    """

    array = pixels.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.permute(1, 2, 0).numpy()).save(path)


@torch.no_grad()
def encode_pixels(vae: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Encode ``[-1, 1]`` pixels to packed, whitened DiT latents.

    Args:
        vae: ``AutoencoderKLFlux2`` in eval mode.
        pixels: ``[batch, 3, height, width]`` on the VAE device.

    Returns:
        ``[batch, 128, height // 16, width // 16]``.
    """

    posterior = vae.encode(pixels.to(device=next(vae.parameters()).device, dtype=vae.dtype)).latent_dist
    packed = patchify_latents(posterior.mode())
    return normalize_patched(packed, vae.bn.running_mean, vae.bn.running_var, float(vae.config.batch_norm_eps))


@torch.no_grad()
def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode DiT latents to pixels in ``[-1, 1]``.

    Args:
        vae: ``AutoencoderKLFlux2`` in eval mode.
        latents: Packed whitened latents ``[batch, 128, height, width]``.

    Returns:
        ``[batch, 3, pixel_height, pixel_width]`` clamped to ``[-1, 1]``.
    """

    native = denormalize_patched(latents.to(vae.dtype), vae.bn.running_mean, vae.bn.running_var, float(vae.config.batch_norm_eps))
    decoded = vae.decode(unpatchify_latents(native), return_dict=False)[0]
    return decoded.clamp(-1, 1)
