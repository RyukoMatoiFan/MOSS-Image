"""Load and save MOSS-Image transformer weights without ComfyUI.

Released bundles store the DiT under ``transformer/`` as a single safetensors file
or a Diffusers index. ``rope.bands`` is a derived constant: a missing or stale
value is rebuilt from ``rope_base``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from training.model import MOSSImageTransformer


def transformer_directory(path: Path) -> Path:
    """Return the directory that contains the DiT weights.

    Args:
        path: A bundle root (with a ``transformer`` child) or the transformer directory itself.

    Returns:
        Directory containing ``config.json`` or ``diffusion_pytorch_model*.safetensors``.
    """

    path = path.expanduser().resolve()
    if (path / "transformer").is_dir():
        return path / "transformer"
    return path


def load_tensor_file(path: Path) -> dict[str, torch.Tensor]:
    """Load one safetensors file or a Diffusers shard index.

    Args:
        path: A ``.safetensors`` file or a directory.

    Returns:
        State dict. Shard paths must stay inside the directory.
    """

    path = path.expanduser().resolve()
    if path.is_file():
        return load_file(str(path))
    index = path / "diffusion_pytorch_model.safetensors.index.json"
    single = path / "diffusion_pytorch_model.safetensors"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for filename in sorted(set(mapping.values())):
            shard = (path / filename).resolve()
            if not shard.is_relative_to(path):
                raise ValueError(f"Shard path escapes the model directory: {filename}")
            state.update(load_file(str(shard)))
        return state
    if single.is_file():
        return load_file(str(single))
    raise FileNotFoundError(f"No diffusion_pytorch_model weights in {path}")


def architecture_from_state(state: dict[str, torch.Tensor]) -> dict[str, float | int]:
    """Read the DiT shape from weight tensors.

    Args:
        state: Unprefixed transformer weights.

    Returns:
        Constructor arguments for ``MOSSImageTransformer``. ``rope_base`` stays 10000.
    """

    required = ("video_embedder.bias", "layers.0.q_norm.weight", "layers.0.proj.ffn_up_gate.weight",
                "final_linear_video.weight", "text_embedder.proj.weight", "time_embedding.0.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Not a MOSS-Image transformer, missing: {missing}")
    dim = int(state["video_embedder.bias"].shape[0])
    head = int(state["layers.0.q_norm.weight"].shape[0])
    layers = 0
    while f"layers.{layers}.norm1.weight" in state:
        layers += 1
    return {
        "dim": dim,
        "num_heads": dim // head,
        "num_layers": layers,
        "ffn_dim": int(state["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2),
        "latent_channels": int(state["final_linear_video.weight"].shape[0]),
        "text_dim": int(state["text_embedder.proj.weight"].shape[1]),
        "freq_dim": int(state["time_embedding.0.weight"].shape[1]),
        "rope_base": 10000.0,
    }


def _strip_prefix(state: dict[str, torch.Tensor], model_keys: set[str]) -> dict[str, torch.Tensor]:
    """Drop a common prefix when the checkpoint uses one and the module does not."""

    if model_keys <= set(state):
        return {key: state[key] for key in model_keys if key in state}
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        stripped = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if model_keys <= set(stripped):
            return {key: stripped[key] for key in model_keys}
    return state


def _split_comfy_qkv(state: dict[str, torch.Tensor]) -> None:
    """Convert native ComfyUI joint-block QKV weights into released model keys.

    The fused projection stores Q, K, V consecutively on its output axis.
    Chunk views avoid allocating another checkpoint-sized copy.
    """

    for key in list(state):
        parts = key.split(".")
        if (len(parts) != 5 or parts[0] != "layers" or not parts[1].isdigit()
                or parts[2:] != ["proj", "qkv", "weight"]):
            continue
        norm_key = f"layers.{parts[1]}.norm1.weight"
        if norm_key not in state or state[norm_key].ndim != 1:
            raise ValueError(f"Cannot infer fused QKV width without {norm_key}")
        dim = state[norm_key].shape[0]
        fused = state[key]
        if fused.shape != (3 * dim, dim):
            raise ValueError(f"Invalid fused QKV shape for {key}: {tuple(fused.shape)}")
        names = [key.replace(".qkv.", f".{projection}.") for projection in ("q", "k", "v")]
        if any(name in state for name in names):
            raise ValueError(f"Checkpoint contains both fused and separate QKV weights: {key}")
        state.update(zip(names, fused.chunk(3, dim=0)))
        del state[key]


def load_transformer(path: Path, gradient_checkpointing: bool = False,
                     dtype: torch.dtype = torch.float32) -> MOSSImageTransformer:
    """Build a transformer and load a released or fine-tuned checkpoint.

    Args:
        path: Bundle root, transformer directory, or dense ComfyUI DiT safetensors.
        gradient_checkpointing: Recompute joint blocks while training.
        dtype: Parameter dtype. LoRA can load directly in bf16 to avoid a full
            fp32 copy of the frozen checkpoint on the host.

    Returns:
        Transformer on CPU in the requested dtype (fp32 by default), eval mode.
    """

    directory = transformer_directory(path)
    state = load_tensor_file(directory)
    # Single-file ComfyUI models need their prefix removed before shape inference.
    if "video_embedder.bias" not in state:
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
            if prefix + "video_embedder.bias" in state:
                state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
                break
    _split_comfy_qkv(state)
    config_path = (directory.parent if directory.is_file() else directory) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architecture = {
            "dim": int(config["dim"]),
            "ffn_dim": int(config["ffn_dim"]),
            "num_heads": int(config["num_heads"]),
            "num_layers": int(config["num_layers"]),
            "text_dim": int(config.get("text_dim", 4096)),
            "freq_dim": int(config.get("freq_dim", 256)),
            "latent_channels": int(config.get("latent_channels", 128)),
            "rope_base": float(config.get("rope_base", 10000.0)),
        }
    else:
        architecture = architecture_from_state(state)
    # Allocate only parameter metadata. A randomly initialized fp32 model would
    # otherwise coexist with both the checkpoint and its fp32 conversion.
    with torch.device("meta"):
        model = MOSSImageTransformer(**architecture, gradient_checkpointing=gradient_checkpointing)
    required = set(model.state_dict()) - {"rope.bands"}
    aligned = _strip_prefix(state, required)
    missing = sorted(required - set(aligned))
    if missing:
        raise ValueError(f"Checkpoint is missing transformer weights: {missing[:8]}")
    unsupported = [key for key in required if aligned[key].dtype not in
                   {torch.float32, torch.float16, torch.bfloat16}]
    if unsupported:
        raise ValueError("Training requires dense fp32/fp16/bf16 weights; packed/FP8 checkpoints "
                         f"need a supported dequantization step first: {unsupported[:4]}")
    del state
    cast = {key: aligned.pop(key).to(device="cpu", dtype=dtype) for key in required}
    incompatible = model.load_state_dict(cast, strict=False, assign=True)
    leftover = [key for key in incompatible.missing_keys if key != "rope.bands"]
    if leftover:
        raise ValueError(f"Failed to load transformer weights: {leftover[:8]}")
    model.rope.bands = model.rope.default_bands(torch.device("cpu"))
    model.eval()
    return model


def save_transformer(model: MOSSImageTransformer, path: Path, *, comfy_layout: bool = False) -> None:
    """Write a full transformer state dict.

    LoRA wrappers are not expected. Call this with the base module for a full
    fine-tune. ``rope.bands`` is included and rebuilt on the next load.

    Args:
        model: Transformer whose ``state_dict`` is the checkpoint.
        path: Destination ``.safetensors`` file.
        comfy_layout: Fuse joint-block Q/K/V weights for native ComfyUI loaders.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if any(".lora_A" in key or ".lora_B" in key or ".base." in key for key in model.state_dict()):
        raise ValueError("Full transformer export requires a base model without LoRA wrappers.")
    from training.safetensors_stream import save_stream

    state = model.state_dict()
    entries = {key: (list(value.shape), [value]) for key, value in state.items()}
    if comfy_layout:
        for index in range(model.num_layers):
            stem = f"layers.{index}.proj."
            pieces = [entries.pop(stem + projection + ".weight")[1][0] for projection in ("q", "k", "v")]
            shape = list(pieces[0].shape)
            shape[0] = sum(piece.shape[0] for piece in pieces)
            entries[stem + "qkv.weight"] = (shape, pieces)
    save_stream(entries, path, metadata={
        "format": "pt", "model": "moss-image", "layout": "comfy" if comfy_layout else "released",
    })
