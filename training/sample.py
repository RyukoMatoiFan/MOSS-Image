"""Sample from a MOSS-Image base checkpoint, a LoRA, or a full fine-tune.

Edit sampling passes clean reference latents with reference indices 1, 2, ...
Both the positive and negative branches see those references, so classifier-free
guidance contrasts the instruction. Negative-branch skipping defaults to block 22,
matching the released pipeline. Shift defaults to 5 and is snapped onto the
training grid.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
class _LazyAutoencoderKLFlux2:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from diffusers import AutoencoderKLFlux2
        return AutoencoderKLFlux2.from_pretrained(*args, **kwargs)

AutoencoderKLFlux2 = _LazyAutoencoderKLFlux2
from PIL import Image, ImageOps

from training.images import decode_latents, encode_pixels, load_rgb, save_rgb
from training.latents import fit_dimensions
from training.lora import LoRALinear, inject_lora, load_lora, lora_spec
from training.memory import place_model
from training.model import model_dtype
from training.schedule import MOSSImageScheduler
from training.text import bundle_paths, encode_prompts
from training.weights import load_transformer

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_states: torch.Tensor,
    negative_states: torch.Tensor,
    references: list[torch.Tensor] | None,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    shift: float,
    skip_layers: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Euler-sample one image.

    Args:
        model: Eval-mode transformer.
        prompt_states: ``[seq, text_dim]`` positive text.
        negative_states: ``[seq, text_dim]`` unconditional text.
        references: Packed reference latents ``[1, 128, h, w]``, or none.
        height: Pixel height, a multiple of 16.
        width: Pixel width, a multiple of 16.
        steps: Requested sampler steps.
        guidance: Classifier-free guidance scale. ``1`` skips the negative branch.
        shift: Inference shift before snapping to the training grid.
        skip_layers: Blocks skipped on the negative branch.
        seed: Generator seed.
        device: Sampling device.

    Returns:
        Latents ``[1, 128, height // 16, width // 16]`` in fp32.
    """

    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        1, model.latent_channels, height // 16, width // 16, generator=generator, dtype=torch.float32,
    ).to(device)
    positive = prompt_states.unsqueeze(0).to(device=device, dtype=model_dtype(model))
    negative = negative_states.unsqueeze(0).to(device=device, dtype=positive.dtype)
    positive_mask = torch.ones(positive.shape[:2], dtype=torch.bool, device=device)
    negative_mask = torch.ones(negative.shape[:2], dtype=torch.bool, device=device)
    refs = None if not references else [ref.to(device=device, dtype=latents.dtype) for ref in references]
    scheduler = MOSSImageScheduler(shift=shift, training_shift=1.0)
    scheduler.set_timesteps(steps, device=device)
    for timestep in scheduler.timesteps:
        conditional = model(latents, timestep, positive, positive_mask, ref_hidden_states=refs).float()
        if guidance == 1:
            prediction = conditional
        else:
            unconditional = model(
                latents, timestep, negative, negative_mask,
                skip_layers=skip_layers, ref_hidden_states=refs,
            ).float()
            prediction = unconditional + guidance * (conditional - unconditional)
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    return latents


def main(argv: list[str] | None = None) -> None:
    """Command line for text-to-image and edit sampling."""

    parser = argparse.ArgumentParser(description="Sample MOSS-Image with an optional LoRA or full fine-tune.")
    parser.add_argument("--bundle", required=True, type=Path, help="Base MOSS-Image bundle.")
    parser.add_argument("--text-encoder", type=Path, help="Optional Comfy text-encoder weights; bundle supplies config/tokenizer.")
    parser.add_argument("--vae", type=Path, help="Optional Comfy VAE weights; bundle supplies base config.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lora", type=Path, help="LoRA safetensors from training.")
    parser.add_argument("--transformer", type=Path, help="Full fine-tune directory or weights file.")
    parser.add_argument("--ref", action="append", default=[], type=Path, help="Reference image. Repeat for several.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--skip-layers", default="22", help="Comma-separated negative-branch blocks. Empty disables.")
    parser.add_argument("--rank", type=int, default=0, help="LoRA rank. 0 reads it from the adapter file.")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha. Defaults to the adapter file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks-to-swap", type=int, default=0, help="Keep the last N transformer blocks on CPU.")
    parser.add_argument("--block-swap-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--base-quantization", choices=("none", "int8", "int8_convrot"), default="none",
                        help="Optional approximate frozen linear-weight storage.")
    parser.add_argument("--convrot-group-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(args.device)
    tokenizer_path, text_path, vae_path = bundle_paths(args.bundle)
    positive, negative = encode_prompts([args.prompt, args.negative], tokenizer_path, text_path, device,
                                       text_encoder_weights=args.text_encoder)
    weight_path = args.transformer if args.transformer is not None else args.bundle
    model = load_transformer(weight_path, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    if args.lora is not None:
        file_rank, file_alpha, targets = lora_spec(args.lora)
        model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        inject_lora(model, args.rank or file_rank, file_alpha, targets)
        load_lora(model, args.lora)
        if args.alpha is not None:
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.alpha = args.alpha
                    module.scale = args.alpha / module.rank
    else:
        if args.alpha is not None:
            logger.warning("--alpha has no effect without --lora; the base model is unchanged.")
        if device.type == "cuda":
            model.to(dtype=torch.bfloat16)
    model.requires_grad_(False).eval()
    if args.base_quantization == "int8":
        from training.quantization import quantize_frozen_linears

        quantize_frozen_linears(model)
    elif args.base_quantization == "int8_convrot":
        from training.convrot import quantize_frozen_convrot_linears

        quantize_frozen_convrot_linears(model, group_size=args.convrot_group_size)
    if args.vae is None:
        vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=torch.float32).eval().to(device)
    else:
        from training.vae import load_vae

        vae = load_vae(vae_path, device, args.vae)
    references = []
    budget = args.height * args.width
    for ref_path in args.ref:
        with Image.open(ref_path) as image:
            ref_w, ref_h = fit_dimensions(*ImageOps.exif_transpose(image).size, budget)
        pixels = load_rgb(ref_path, ref_w, ref_h).unsqueeze(0).to(device)
        references.append(encode_pixels(vae, pixels))
    vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.block_swap_mode == "async":
        place_model(model, device, args.blocks_to_swap, block_swap_mode="async")
    else:
        place_model(model, device, args.blocks_to_swap)
    skip_layers = tuple(int(item) for item in args.skip_layers.split(",") if item.strip())
    latents = generate(
        model, positive, negative, references or None,
        args.height, args.width, args.steps, args.guidance, args.shift,
        skip_layers, args.seed, device,
    )
    place_model(model, torch.device("cpu"), 0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vae.to(device)
    image = decode_latents(vae, latents.to(vae.dtype))[0]
    save_rgb(image, args.out)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
