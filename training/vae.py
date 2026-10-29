"""Strict Flux.2 VAE loading from Diffusers bundles or Comfy safetensors."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import torch


def _original_key(key: str, up_blocks: int) -> str:
    """Map one Diffusers key into the original Flux.2/Comfy state layout."""
    key = key.replace("encoder.conv_norm_out.", "encoder.norm_out.")
    key = key.replace("decoder.conv_norm_out.", "decoder.norm_out.")
    if key.startswith("quant_conv."):
        return "encoder." + key
    if key.startswith("post_quant_conv."):
        return "decoder." + key
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", key)
    key = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.",
                 lambda m: f"decoder.up.{up_blocks - 1 - int(m[1])}.block.{m[2]}.", key)
    key = re.sub(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
                 lambda m: f"decoder.up.{up_blocks - 1 - int(m[1])}.upsample.", key)
    key = re.sub(r"\.mid_block\.resnets\.(\d+)\.", lambda m: f".mid.block_{int(m[1]) + 1}.", key)
    key = key.replace("conv_shortcut.", "nin_shortcut.")
    key = key.replace(".mid_block.attentions.0.", ".mid.attn_1.")
    for source, target in (("group_norm", "norm"), ("to_q", "q"), ("to_k", "k"),
                           ("to_v", "v"), ("to_out.0", "proj_out")):
        key = key.replace(f".attn_1.{source}.", f".attn_1.{target}.")
    return key


def _convert_state(state, expected, up_blocks):
    original = any(".mid.attn_1." in key or ".down.0.block." in key for key in state)
    result = {}
    consumed = set()
    for key, target in expected.items():
        source = _original_key(key, up_blocks) if original else key
        if source not in state:
            raise ValueError(f"VAE checkpoint is missing {source} (required for {key}).")
        value = state[source]
        if original and ".attn_1." in source and value.ndim == 4 and value.shape[-2:] == (1, 1):
            value = value[:, :, 0, 0]
        if value.shape != target.shape:
            raise ValueError(f"VAE shape mismatch for {source}: {tuple(value.shape)} vs {tuple(target.shape)}.")
        if value.is_floating_point() != target.is_floating_point():
            raise ValueError(f"VAE dtype category mismatch for {source}.")
        result[key] = value.to(dtype=target.dtype)
        consumed.add(source)
    extra = sorted(set(state) - consumed)
    if extra:
        raise ValueError(f"Unrecognized VAE checkpoint tensors: {extra[:5]}")
    return result


def load_vae(config_path: Path, device, weights_path: Path | None = None):
    """Load an fp32 evaluation VAE, validating every single-file tensor strictly.

    ``config_path`` is a Diffusers VAE directory or config.json. A single-file
    Comfy decoder widths are inferred from its block weights with an explicit
    warning when they override the config. Every other architecture field is kept.
    No missing weights are initialized.
    """
    from diffusers import AutoencoderKLFlux2

    config_path = Path(config_path)
    directory = config_path.parent if config_path.is_file() else config_path
    if weights_path is None:
        model = AutoencoderKLFlux2.from_pretrained(str(directory), torch_dtype=torch.float32)
    else:
        from safetensors.torch import load_file

        config = AutoencoderKLFlux2.load_config(str(directory))
        state = load_file(str(weights_path), device="cpu")
        if "decoder.up.0.block.0.conv1.weight" in state:
            widths = [state[f"decoder.up.{index}.block.0.conv1.weight"].shape[0]
                      for index in range(len(config["up_block_types"]))]
            configured = config.get("decoder_block_out_channels") or config["block_out_channels"]
            if list(configured) != widths:
                warnings.warn(f"Comfy VAE decoder widths {widths} override config widths {list(configured)}; "
                              "all converted tensors will be strictly validated.", UserWarning, stacklevel=2)
                config["decoder_block_out_channels"] = widths
        with torch.device("meta"):
            model = AutoencoderKLFlux2.from_config(config)
        state = _convert_state(state, model.state_dict(), len(model.decoder.up_blocks))
        model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
