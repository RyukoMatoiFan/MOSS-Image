"""Prompt encoding that follows the released MOSS-Image pipeline.

Each caption is rendered with the Qwen chat template, thinking enabled, and
encoded alone so padding cannot change the hidden states. The text encoder stays
frozen. Training reads the cached last-layer states and does not load this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_text_encoder(config_path: Path, device: torch.device, dtype: torch.dtype,
                      weights_path: Path | None = None) -> torch.nn.Module:
    """Load bundle weights or decode a Comfy single-file encoder for precaching.

    Comfy INT8 weights expand to dense parameters here. Cached embeddings therefore
    use floating-point inference, without Comfy's dynamic activation quantization.
    """
    if weights_path is None:
        return AutoModel.from_pretrained(config_path, dtype=dtype).eval().to(device)
    from training.comfy_weights import load_comfy_state

    state = load_comfy_state(weights_path, dtype=dtype)
    if "model.language_model.embed_tokens.weight" in state:
        state = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    config = AutoConfig.from_pretrained(config_path)
    # Resolve the public AutoModel mapping without allocating full parameters.
    # Its from_pretrained factory does not accept None in some Transformers
    # versions, while the concrete class supports state_dict-only loading.
    with torch.device("meta"):
        template = AutoModel.from_config(config)
    encoder_class = type(template)
    del template
    encoder, information = encoder_class.from_pretrained(
        None, config=config, state_dict=state, dtype=dtype, output_loading_info=True,
    )
    if information.get("missing_keys") or information.get("unexpected_keys") or information.get("mismatched_keys"):
        raise ValueError(f"Comfy text encoder does not match its configuration: {information}")
    return encoder.eval().to(device)


def bundle_paths(bundle: Path) -> tuple[Path, Path, Path]:
    """Locate the tokenizer, text encoder, and VAE inside a MOSS-Image bundle.

    Args:
        bundle: Directory that contains ``tokenizer``, ``text_encoder``, and ``vae``,
            or a transformer directory whose parent is that bundle.

    Returns:
        ``(tokenizer, text_encoder, vae)`` directories.
    """

    root = bundle.expanduser().resolve()
    if not (root / "text_encoder").is_dir() and (root.parent / "text_encoder").is_dir():
        root = root.parent
    tokenizer = root / "tokenizer"
    text_encoder = root / "text_encoder"
    vae = root / "vae"
    missing = [str(path) for path in (tokenizer, text_encoder, vae) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"MOSS-Image bundle is missing: {missing}")
    return tokenizer, text_encoder, vae


@torch.no_grad()
def iter_encode_prompts(
    prompts: list[str],
    tokenizer_path: Path,
    text_encoder_path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 4096,
    text_encoder_weights: Path | None = None,
) -> Iterator[torch.Tensor]:
    """Encode captions the way ``MOSSImagePipeline.encode_prompt`` does.

    Args:
        prompts: Caption strings. An empty string is the unconditional prompt.
        tokenizer_path: Tokenizer directory from the bundle.
        text_encoder_path: Qwen3.5 text-encoder directory.
        device: Device for the encoder forward.
        dtype: Encoder dtype. fp32 is used when ``device`` is CPU.
        max_sequence_length: Truncation length, from 1 to 4096.

    Returns:
        One ``[seq, text_dim]`` CPU tensor per prompt, with padding removed.
    """

    if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a nonempty list of strings.")
    if not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be between 1 and 4096.")
    if device.type == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encoder = load_text_encoder(text_encoder_path, device, dtype, text_encoder_weights)
    try:
        for prompt in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
            yield hidden.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    finally:
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()


def encode_prompts(
    prompts: list[str], tokenizer_path: Path, text_encoder_path: Path,
    device: torch.device, dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 4096,
    text_encoder_weights: Path | None = None,
) -> list[torch.Tensor]:
    """Collect prompt states; precaching uses the streaming variant instead."""
    return list(iter_encode_prompts(prompts, tokenizer_path, text_encoder_path,
                                   device, dtype, max_sequence_length, text_encoder_weights))
