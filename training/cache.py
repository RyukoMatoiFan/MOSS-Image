"""On-disk cache of patched latents and frozen text states.

Each example is a safetensors file plus a JSON sidecar. Reference latents, when
present, are ``ref_0``, ``ref_1``, and so on. An ``empty`` example holds the
unconditional prompt used for classifier-free guidance dropout.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def write_example(
    path: Path,
    latents: torch.Tensor,
    context: torch.Tensor,
    refs: list[torch.Tensor],
    prompt: str,
    height: int,
    width: int,
) -> None:
    """Write one training example.

    Args:
        path: Destination ``.safetensors`` path. The sidecar replaces the suffix with ``.json``.
        latents: Target tokens ``[channels, height, width]``.
        context: Text states ``[seq, text_dim]``.
        refs: Reference token grids, each ``[channels, height, width]``.
        prompt: Caption or edit instruction stored for sampling.
        height: Target height in pixels.
        width: Target width in pixels.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "latents": latents.detach().to(dtype=torch.bfloat16, device="cpu").contiguous(),
        "context": context.detach().to(dtype=torch.bfloat16, device="cpu").contiguous(),
    }
    ref_shapes: list[list[int]] = []
    for index, reference in enumerate(refs):
        tensors[f"ref_{index}"] = reference.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
        ref_shapes.append(list(reference.shape))
    save_file(tensors, str(path))
    sidecar = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "latent": list(latents.shape),
        "context": list(context.shape),
        "refs": ref_shapes,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")


def read_example(path: Path, *, copy_tensors: bool = False) -> dict[str, object]:
    """Load one example.

    Args:
        path: ``.safetensors`` path written by ``write_example``.
        copy_tensors: Release file mappings before returning, allowing the file
            to be overwritten during streamed precaching on Windows.

    Returns:
        Dict with ``latents``, ``context``, ``refs``, ``prompt``, ``height``, and ``width``.
        Tensors are CPU bf16.
    """

    tensors = load_file(str(path))
    if copy_tensors:
        tensors = {key: value.clone() for key, value in tensors.items()}
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    refs = [tensors[f"ref_{index}"] for index in range(len(sidecar["refs"]))]
    return {
        "latents": tensors["latents"],
        "context": tensors["context"],
        "refs": refs,
        "prompt": sidecar["prompt"],
        "height": int(sidecar["height"]),
        "width": int(sidecar["width"]),
    }


def write_index(cache_dir: Path, rows: list[dict[str, object]]) -> None:
    """Write ``index.jsonl`` for the training loop.

    Args:
        cache_dir: Cache directory.
        rows: One JSON object per training example. ``empty`` is not included.
    """

    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    (cache_dir / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_index(cache_dir: Path) -> list[dict[str, object]]:
    """Read ``index.jsonl``.

    Args:
        cache_dir: Cache directory produced by precache.

    Returns:
        Index rows. Each row has a ``file`` field naming a safetensors file.
    """

    path = cache_dir / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache index: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Cache index is empty: {path}")
    return rows


def _file_identity(path: Path) -> dict:
    """Hash file contents in bounded chunks; used once during cache construction."""
    if not path.is_file():
        return {"path": str(path.resolve()), "missing": True}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _component_identity(path: Path) -> dict:
    if not path.is_dir():
        return _file_identity(path)
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken", ".jinja", ".jinja2"}
    files = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix in suffixes)
    return {"path": str(path.resolve()), "files": [_file_identity(item) for item in files]}


def _fingerprint(settings: dict) -> str:
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_provenance(tokenizer: Path, text_encoder: Path, vae: Path, jobs: list[dict], resolution: int,
                     *, text_config: Path, vae_config: Path, text_dtype: str = "unspecified") -> dict:
    """Record model/source content identities and preprocessing conventions."""
    settings = {
        "version": 1,
        "preprocessing": {"resolution": resolution, "orientation": "exif", "resize": "lanczos-area-stride16",
                          "posterior": "mode", "pack": [2, 2], "normalization": "vae-running-bn",
                          "thinking": True, "max_sequence_length": 4096, "storage": "bfloat16",
                          "text_dtype": text_dtype, "vae_dtype": "float32"},
        "libraries": {name: metadata.version(name) for name in ("torch", "transformers", "diffusers")},
        "tokenizer": _component_identity(tokenizer), "text_encoder": _component_identity(text_encoder),
        "vae": _component_identity(vae),
        "text_config": _file_identity(text_config / "config.json"),
        "vae_config": _file_identity(vae_config / "config.json"),
        "sources": [{"target": _file_identity(Path(job["target"])), "prompt": str(job["prompt"]),
                     "refs": [_file_identity(Path(path)) for path in job["refs"]]} for job in jobs],
    }
    return {"fingerprint": _fingerprint(settings), "settings": settings}


def write_provenance(cache_dir: Path, provenance: dict) -> None:
    (cache_dir / "cache_info.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_fingerprint(cache_dir: Path, expected: str = "") -> str:
    """Validate manifest integrity and an optional expected cache identity."""
    manifest = cache_dir / "cache_info.json"
    if not manifest.is_file():
        if expected:
            raise ValueError("Cache has no fingerprint metadata; rebuild it or choose a matching cache.")
        return ""
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    actual = _fingerprint(contents["settings"])
    if actual != contents["fingerprint"] or (expected and actual != expected):
        raise ValueError("Cache fingerprint does not match its metadata or the expected cache identity.")
    return actual
