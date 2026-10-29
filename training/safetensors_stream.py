"""Safetensors serialization with at most one tensor copied to the host at a time."""

from __future__ import annotations

import json
import math
import os
import struct
import uuid
from pathlib import Path

import torch

_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint8: "U8", torch.bool: "BOOL",
}


def save_stream(
    entries: dict[str, tuple[list[int], list[torch.Tensor]]],
    path: Path,
    metadata: dict[str, str],
) -> None:
    """Write entries; multiple pieces concatenate on dim zero without a fused copy."""
    header: dict = {"__metadata__": metadata}
    offset = 0
    for name, (shape, pieces) in entries.items():
        dtype = pieces[0].dtype
        if dtype not in _DTYPES or any(piece.dtype != dtype for piece in pieces):
            raise ValueError(f"Unsupported or inconsistent checkpoint dtype for {name}.")
        size = sum(piece.numel() * piece.element_size() for piece in pieces)
        if size != math.prod(shape) * pieces[0].element_size():
            raise ValueError(f"Checkpoint shape does not match tensor data for {name}.")
        header[name] = {"dtype": _DTYPES[dtype], "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for _, pieces in entries.values():
                for piece in pieces:
                    host = piece.detach().to(device="cpu").contiguous()
                    handle.write(memoryview(host.reshape(-1).view(torch.uint8).numpy()))
                    del host
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
