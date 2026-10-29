"""LoRA, full fine-tune, and reference-edit training for MOSS-Image.

The DiT matches the released ``pipeline_moss_image.py`` state dict and text-to-image
forward. Edit training packs clean reference latents on the reserved RoPE reference
axis. The Qwen3.5-9B text encoder and the FLUX.2 VAE stay frozen and are used only
while precaching or sampling.
"""
