"""Cache patched latents and frozen Qwen3.5 states for MOSS-Image training.

Text-to-image data is a directory of images with a ``.txt`` caption beside each
file. Edit data is a JSONL manifest::

    {"target": "after.png", "refs": ["before.png"], "instruction": "make it autumn"}

The text encoder and VAE are loaded one at a time. The training loop never loads them.
"""

from __future__ import annotations

import argparse
import json
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
from safetensors.torch import save_file

from training.cache import cache_provenance, read_example, write_example, write_index, write_provenance
from training.images import encode_pixels, load_rgb
from training.latents import fit_dimensions
from training.text import bundle_paths, iter_encode_prompts

logger = logging.getLogger(__name__)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _jobs_from_images(root: Path) -> list[dict[str, object]]:
    """Pair each image with a same-stem caption file.

    Args:
        root: Directory of images. Captions are ``<stem>.txt``.

    Returns:
        Jobs with ``target``, ``refs``, and ``prompt``.
    """

    jobs = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        caption_path = path.with_suffix(".txt")
        if not caption_path.is_file():
            raise FileNotFoundError(f"Missing caption for {path.name}: {caption_path}")
        jobs.append({"target": path, "refs": [], "prompt": caption_path.read_text(encoding="utf-8").strip()})
    if not jobs:
        raise ValueError(f"No images found in {root}")
    return jobs


def _jobs_from_manifest(path: Path) -> list[dict[str, object]]:
    """Read an edit or caption manifest.

    Args:
        path: JSONL file. Each row has ``target`` and either ``instruction`` or ``caption``.
            ``refs`` is an optional list of image paths. Relative paths are based at the
            manifest's directory.

    Returns:
        Jobs with absolute paths.
    """

    jobs = []
    root = path.parent
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "target" not in row:
            raise ValueError(f"{path}:{line_number} is missing target")
        prompt = row.get("instruction", row.get("caption"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{line_number} needs instruction or caption text")
        refs = [root / item if not Path(item).is_absolute() else Path(item) for item in row.get("refs", [])]
        target = Path(row["target"])
        jobs.append({
            "target": target if target.is_absolute() else root / target,
            "refs": refs,
            "prompt": prompt.strip(),
        })
    if not jobs:
        raise ValueError(f"Manifest is empty: {path}")
    return jobs


def _encode_image(vae: torch.nn.Module, path: Path, max_pixels: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Resize one image onto the pixel budget and encode it.

    Args:
        vae: FLUX.2 VAE.
        path: Image path.
        max_pixels: Pixel budget.
        device: VAE device.

    Returns:
        Packed latent ``[128, grid_h, grid_w]``, pixel height, and pixel width.
    """

    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    width, height = fit_dimensions(width, height, max_pixels)
    pixels = load_rgb(path, width, height).unsqueeze(0).to(device)
    latent = encode_pixels(vae, pixels)[0].cpu()
    return latent, height, width


def precache(bundle: Path, output: Path, jobs: list[dict[str, object]], resolution: int,
             device: torch.device, limit: int, *, text_encoder_weights: Path | None = None,
             vae_weights: Path | None = None) -> None:
    """Encode ``jobs`` into ``output``.

    Args:
        bundle: MOSS-Image directory with tokenizer, text encoder, and VAE.
        output: Cache directory.
        jobs: Caption or edit jobs.
        resolution: Pixel budget is ``resolution ** 2``.
        device: Device for the VAE and, afterwards, the text encoder.
        limit: If positive, keep only the first jobs.
    """

    if limit > 0:
        jobs = jobs[:limit]
    if (output / "index.jsonl").exists():
        raise FileExistsError(f"Cache already exists: {output}. Choose a new output directory.")
    tokenizer_path, text_path, vae_path = bundle_paths(bundle)
    max_pixels = resolution * resolution
    provenance = cache_provenance(tokenizer_path, text_encoder_weights or text_path,
                                  vae_weights or vae_path, jobs, resolution,
                                  text_config=text_path, vae_config=vae_path,
                                  text_dtype="float32" if device.type == "cpu" else "bfloat16")
    if vae_weights is None:
        vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=torch.float32).eval().to(device)
    else:
        from training.vae import load_vae

        vae = load_vae(vae_path, device, vae_weights)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index, job in enumerate(jobs):
            target, height, width = _encode_image(vae, Path(job["target"]), max_pixels, device)
            refs = []
            for ref_path in job["refs"]:
                ref_latent, _, _ = _encode_image(vae, Path(ref_path), max_pixels, device)
                refs.append(ref_latent)
            name = f"{index:06d}.safetensors"
            write_example(output / name, target, torch.empty(0, 0), refs,
                          str(job["prompt"]), height, width)
            rows.append({"file": name, "latent": list(target.shape),
                         "refs": [list(ref.shape) for ref in refs], "height": height, "width": width})
            logger.info("encoded image %s/%s", index + 1, len(jobs))
    finally:
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prompts = [str(job["prompt"]) for job in jobs]
    contexts = iter_encode_prompts(prompts + [""], tokenizer_path, text_path, device,
                                   text_encoder_weights=text_encoder_weights)
    for index, context in enumerate(contexts):
        if index == len(jobs):
            save_file({"context": context.contiguous()}, str(output / "empty.safetensors"))
            continue
        name = str(rows[index]["file"])
        item = read_example(output / name, copy_tensors=True)
        write_example(
            output / name,
            item["latents"],
            context,
            item["refs"],
            str(item["prompt"]),
            int(item["height"]),
            int(item["width"]),
        )
        rows[index]["context_tokens"] = int(context.shape[0])
    write_provenance(output, provenance)
    write_index(output, rows)
    logger.info("wrote %s examples to %s", len(rows), output)


def main(argv: list[str] | None = None) -> None:
    """Command line for precaching."""

    parser = argparse.ArgumentParser(description="Precache MOSS-Image latents and text states.")
    parser.add_argument("--bundle", required=True, type=Path, help="Local MOSS-Image bundle.")
    parser.add_argument("--text-encoder", type=Path, help="Optional Comfy text-encoder safetensors; bundle supplies config/tokenizer.")
    parser.add_argument("--vae", type=Path, help="Optional Comfy VAE safetensors; bundle supplies base config.")
    parser.add_argument("--out", required=True, type=Path, help="Cache directory.")
    parser.add_argument("--images", type=Path, help="Directory of images with .txt captions.")
    parser.add_argument("--manifest", type=Path, help="JSONL edit or caption manifest.")
    parser.add_argument("--resolution", type=int, default=1024, help="Pixel budget side length.")
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N examples.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if (args.images is None) == (args.manifest is None):
        raise SystemExit("Provide exactly one of --images and --manifest.")
    jobs = _jobs_from_images(args.images) if args.images is not None else _jobs_from_manifest(args.manifest)
    precache(args.bundle, args.out, jobs, args.resolution, torch.device(args.device), args.limit,
             text_encoder_weights=args.text_encoder, vae_weights=args.vae)


if __name__ == "__main__":
    main()
