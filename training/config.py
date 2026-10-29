"""YAML configuration for the MOSS-Image trainers.

Machine-specific paths belong in the YAML. The dataclass defaults are enough for
a smoke run. Environment variables are not read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    """Training settings.

    Attributes:
        model: Bundle directory, transformer directory, or dense single-file checkpoint.
        cache_dir: Precache output.
        output_dir: Checkpoints and ``metrics.jsonl``.
        mode: ``lora`` or ``full``.
        resolution: Pixel budget is ``resolution ** 2``. Training rejects cached
            images and references above that area and writes the value into
            ``train_config.json``. Precache takes the same budget on its own
            ``--resolution`` flag.
        cfg_dropout_prob: Probability of replacing the caption with the empty prompt.
        ref_dropout_prob: Probability of dropping reference latents on an edit example.
        n_eval_holdout: Last index rows reserved for evaluation loss.
        eval_every: Completed updates between evaluations; 0 follows ckpt_every.
        eval_sigmas: Fixed validation noise levels; every level uses eval_seed.
        eval_seed: Independent validation noise seed.
        rank: LoRA rank. Ignored for full fine-tuning.
        alpha: LoRA alpha. Ignored for full fine-tuning. After a LoRA resume,
            a single alpha stored in the adapter replaces this value so the
            recorded config matches the scale that actually trains.
        targets: ``joint`` excludes the text refiner; ``joint_and_text`` includes it.
            Legacy ``blocks`` includes text; ``blocks_and_io`` also includes image I/O.
        lr: Learning rate.
        steps: Optimizer steps.
        batch: Examples per step. Matching latent and reference shapes share a
            forward. The loss is the mean over target tokens, so a larger image
            weighs more than a smaller one.
        accum: Gradient accumulation steps. The optimizer step still averages
            over every target element, so a larger micro-batch weighs more.
        max_tokens: Per-microbatch target + reference + text token cap; 0 disables.
        packed_batching: Execute heterogeneous examples in one segmented token graph.
        warmup: Linear warmup steps.
        optimizer: ``adamw``, ``adamw8bit``, native PyTorch ``adafactor``, or the
            absolute-LR, in-backward ``adafactor_fused`` for full training.
        weight_decay: AdamW weight decay.
        adam_betas: AdamW first/second moment coefficients.
        weight_decay_policy: ``all`` preserves blanket decay; ``matrix`` excludes
            norms, biases, vectors and input embeddings.
        qk_norm_decay_to_one: Decay the last joint block's Q/K norm toward one.
        grad_clip: Global grad-norm clip. ``0`` disables clipping.
        grad_checkpointing: Recompute joint blocks.
        checkpoint_fraction: Deterministic joint-block checkpoint share in [0,1].
            Swapped blocks always recompute regardless of this setting.
        activation_offload: Offload autograd-saved CUDA activations to pinned CPU memory.
        gradient_accumulation_device: ``cpu`` stores early microbatch gradients in
            host FP32 buffers for fused full training; ``device`` retains them locally.
        blocks_to_swap: Last N joint blocks keep CPU master weights and optimizer
            states. Synchronous checkpointed device copies support LoRA and full
            training with AdamW or Adafactor; requires host RAM and PCIe traffic.
        block_swap_mode: ``sync`` uses differentiable CPU-master copies;
            ``async`` streams frozen base tensors from pinned CPU memory through
            two GPU staging slots while LoRA adapters/optimizer remain on CUDA.
        mixed_precision: ``no`` or ``bf16`` autocast. Full training retains fp32
            master parameters and accumulates fp32 gradients.
        attention_backend: ``auto``/``sdpa`` use native PyTorch SDPA;
            ``sdpa_cudnn`` explicitly selects cuDNN attention; ``flash_attn_2``
            requires a matching optional FlashAttention package.
        compile_blocks: Compile one shared joint-block implementation. Checkpoint
            and offload orchestration remain outside the compiled function.
        compile_mode: Torch compile mode. ``default`` avoids CUDA graph copies.
        lora_compute_dtype: Optional ``fp32``, ``bf16``, or ``fp16`` adapter
            compute dtype. ``None`` follows autocast while retaining fp32 masters.
        full_parameter_dtype: ``fp32`` or ``bf16`` trainable parameter storage for
            full mode. The trainer enforces the supported optimizer/device pairing.
        base_quantization: ``none``, ``int8`` or ``int8_convrot`` frozen linear
            storage, LoRA only. ConvRot uses grouped normalized Hadamard rotation.
        quantized_backend: ``eager`` rematerializes floating weights; ``triton``
            decodes weight tiles in the GEMM; ``auto`` selects supported CUDA kernels.
        convrot_group_size: Maximum Hadamard group width for ``int8_convrot``.
            Each layer uses a power of two dividing its input width, at most this.
        checkpoint_format: ``released`` stores separate Q/K/V weights; ``comfy``
            fuses them for native ComfyUI full-checkpoint loading. Both can resume.
        min_lr_ratio: Cosine floor as a fraction of ``lr``.
        seed: Seed for Python, NumPy, and PyTorch. CUDA reductions stay nondeterministic.
        flow_total: Training timestep count.
        flow_mu: Logit-normal mean.
        flow_width: Logit-normal standard deviation.
        flow_cutoff: Truncation before rescaling.
        flow_uniform: Uniform mixture weight.
        log_every: Steps between log lines.
        ckpt_every: Steps between checkpoints.
        resume: Checkpoint directory to resume, or empty.
        warm_start: Initialize full weights or a LoRA adapter without optimizer state.
        expected_cache_fingerprint: Optional SHA256 identity required from cache_info.json.
    """

    model: str = ""
    cache_dir: str = "cache"
    output_dir: str = "runs/moss"
    mode: str = "lora"
    resolution: int = 1024
    cfg_dropout_prob: float = 0.1
    ref_dropout_prob: float = 0.0
    n_eval_holdout: int = 0
    eval_every: int = 0
    eval_sigmas: tuple[float, ...] = (0.25, 0.5, 0.75)
    eval_seed: int = 0
    rank: int = 16
    alpha: float = 16.0
    targets: str = "blocks"
    lr: float = 1e-4
    steps: int = 1000
    batch: int = 1
    accum: int = 1
    max_tokens: int = 0
    packed_batching: bool = False
    warmup: int = 50
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay_policy: str = "all"
    qk_norm_decay_to_one: bool = False
    grad_clip: float = 1.0
    grad_checkpointing: bool = True
    checkpoint_fraction: float = 1.0
    activation_offload: bool = False
    gradient_accumulation_device: str = "device"
    blocks_to_swap: int = 0
    block_swap_mode: str = "sync"  # async: pinned frozen bases, CUDA LoRA adapters
    mixed_precision: str = "no"
    attention_backend: str = "auto"
    compile_blocks: bool = False
    compile_mode: str = "default"
    lora_compute_dtype: str | None = None
    full_parameter_dtype: str = "fp32"
    base_quantization: str = "none"
    quantized_backend: str = "eager"
    convrot_group_size: int = 256
    checkpoint_format: str = "released"
    min_lr_ratio: float = 0.1
    seed: int = 0
    flow_total: int = 1000
    flow_mu: float = -1.09
    flow_width: float = 0.95
    flow_cutoff: float = 0.95
    flow_uniform: float = 0.01
    log_every: int = 10
    ckpt_every: int = 200
    resume: str = ""
    warm_start: str = ""
    expected_cache_fingerprint: str = ""

    def output_path(self) -> Path:
        """Return the run directory."""

        return Path(self.output_dir)


def load_config(path: Path) -> TrainConfig:
    """Load a YAML file into ``TrainConfig``.

    Unknown keys raise. Missing keys keep the dataclass default.

    Args:
        path: YAML config.

    Returns:
        Populated config.
    """

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Training configs need PyYAML. Install it with: pip install -e '.[train]'") from exc
    text = path.read_text(encoding="utf-8")
    raw = (json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)) or {}
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a mapping.")
    known = {item.name for item in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    values = {key: raw[key] for key in known if key in raw}
    config = TrainConfig(**values)
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    """Validate YAML and directly constructed configurations before loading weights."""
    if config.mode not in {"lora", "full"}:
        raise ValueError("mode must be 'lora' or 'full'.")
    if config.optimizer not in {"adamw", "adamw8bit", "adafactor", "adafactor_fused"}:
        raise ValueError("optimizer must be 'adamw', 'adamw8bit', 'adafactor', or 'adafactor_fused'.")
    if config.targets not in {"joint", "joint_and_text", "blocks", "blocks_and_io"}:
        raise ValueError("targets must be 'joint', 'joint_and_text', 'blocks', or 'blocks_and_io'.")
    if config.mixed_precision not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be 'no' or 'bf16'.")
    if config.checkpoint_format not in {"released", "comfy"}:
        raise ValueError("checkpoint_format must be 'released' or 'comfy'.")
    if config.base_quantization not in {"none", "int8", "int8_convrot"}:
        raise ValueError("base_quantization must be 'none', 'int8', or 'int8_convrot'.")
    if config.block_swap_mode not in {"sync", "async", "async_full"}:
        raise ValueError("block_swap_mode must be 'sync', 'async', or 'async_full'.")
    if config.blocks_to_swap < 0:
        raise ValueError("blocks_to_swap must be nonnegative.")
    if config.mode == "lora":
        if config.rank < 1:
            raise ValueError("rank must be at least 1.")
        if config.alpha <= 0:
            raise ValueError("alpha must be positive.")
    if config.convrot_group_size < 1:
        raise ValueError("convrot_group_size must be positive.")
    if config.batch < 1 or config.accum < 1 or config.steps < 1:
        raise ValueError("batch, accum, and steps must be positive.")
    if config.resolution < 16:
        raise ValueError("resolution must be at least 16.")
    if config.attention_backend not in {"auto", "sdpa", "sdpa_cudnn", "flash_attn_2"}:
        raise ValueError("attention_backend must be 'auto', 'sdpa', 'sdpa_cudnn', or 'flash_attn_2'.")
    if not isinstance(config.compile_blocks, bool):
        raise ValueError("compile_blocks must be true or false.")
    if not isinstance(config.compile_mode, str) or not config.compile_mode.strip():
        raise ValueError("compile_mode must be a non-empty string.")
    if config.compile_mode in {"reduce-overhead", "max-autotune"}:
        raise ValueError("compile_mode 'reduce-overhead' and 'max-autotune' enable CUDA graphs; use 'default' or 'max-autotune-no-cudagraphs'.")
    if config.lora_compute_dtype not in {None, "fp32", "bf16", "fp16"}:
        raise ValueError("lora_compute_dtype must be null, 'fp32', 'bf16', or 'fp16'.")
    if config.full_parameter_dtype not in {"fp32", "bf16"}:
        raise ValueError("full_parameter_dtype must be 'fp32' or 'bf16'.")
    if not 0 <= config.checkpoint_fraction <= 1:
        raise ValueError("checkpoint_fraction must be between 0 and 1.")
    if config.max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative; 0 disables the token budget.")
    if config.gradient_accumulation_device not in {"device", "cpu"}:
        raise ValueError("gradient_accumulation_device must be 'device' or 'cpu'.")
    if config.gradient_accumulation_device == "cpu" and config.optimizer != "adafactor_fused":
        raise ValueError("CPU gradient accumulation requires optimizer adafactor_fused.")
    if config.quantized_backend not in {"eager", "auto", "triton"}:
        raise ValueError("quantized_backend must be 'eager', 'auto', or 'triton'.")
    if config.weight_decay_policy not in {"all", "matrix"}:
        raise ValueError("weight_decay_policy must be 'all' or 'matrix'.")
    if len(config.adam_betas) != 2 or any(not 0 <= value < 1 for value in config.adam_betas):
        raise ValueError("adam_betas must contain two values in [0, 1).")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative.")
    if config.log_every < 1 or config.ckpt_every < 1 or config.eval_every < 0:
        raise ValueError("log_every/ckpt_every must be positive and eval_every nonnegative.")
    if not config.eval_sigmas or any(not 0 < sigma <= 1 for sigma in config.eval_sigmas):
        raise ValueError("eval_sigmas must be a nonempty sequence in (0, 1].")
    if config.resume and config.warm_start:
        raise ValueError("Use either resume or warm_start, not both.")
    for name in ("packed_batching", "activation_offload", "qk_norm_decay_to_one"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be true or false.")
