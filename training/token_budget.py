"""Token accounting for target, reference images and real text in a microbatch."""

from __future__ import annotations

import math
from pathlib import Path

from safetensors import safe_open


def prepared_tokens(example) -> int:
    target, context, references = example
    return target.shape[-2] * target.shape[-1] + context.shape[0] + sum(
        reference.shape[-2] * reference.shape[-1] for reference in references
    )


def index_tokens(row: dict, cache: Path, empty_length: int) -> int:
    """Conservative row cost including caption dropout's empty-prompt length."""
    with safe_open(str(cache / row["file"]), framework="pt", device="cpu") as reader:
        target = reader.get_slice("latents").get_shape()
        context = reader.get_slice("context").get_shape()
        references = [reader.get_slice(name).get_shape() for name in reader.keys() if name.startswith("ref_")]
    return math.prod(target[-2:]) + max(context[0], empty_length) + sum(math.prod(shape[-2:]) for shape in references)
