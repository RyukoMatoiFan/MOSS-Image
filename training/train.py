"""LoRA and full fine-tuning for MOSS-Image, including reference-image edits.

Examples are read from a precache directory. A row with reference latents trains
the edit objective: those latents stay clean, ``r`` starts at 1, and the loss is
only on the noised target. Rows without references are text-to-image. The text
encoder is not in the optimizer.

CUDA matmul reductions are nondeterministic even with ``seed`` set.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from training.cache import cache_fingerprint, read_example, read_index
from training.config import TrainConfig, load_config, validate_config
from training.flow import FlowMatcher
from training.lora import LoRALinear, inject_lora, load_lora, lora_parameters, save_lora
from training.memory import place_model
from training.model import MOSSImageTransformer, model_dtype
from training.performance import configure_model_performance
from training.weights import load_transformer, save_transformer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch.

    Args:
        seed: Non-negative seed. CUDA atomic adds can still change the exact bits.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_context(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length text states and build a binary mask.

    Args:
        rows: ``[seq, text_dim]`` tensors.

    Returns:
        Embeddings ``[batch, max_seq, text_dim]`` and mask ``[batch, max_seq]``.
    """

    length = max(row.shape[0] for row in rows)
    embeddings = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for index, row in enumerate(rows):
        embeddings[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = True
    return embeddings, mask


def _prepare_example(
    example: dict[str, object],
    empty: torch.Tensor,
    cfg_dropout: float,
    ref_dropout: float,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Apply caption and reference dropout to one cached example.

    Args:
        example: Record from ``read_example``.
        empty: Unconditional text states ``[seq, text_dim]``.
        cfg_dropout: Probability of using ``empty``.
        ref_dropout: Probability of dropping reference latents.

    Returns:
        Target latents, text states, and the reference list used for this step.
    """

    context = example["context"]
    refs = list(example["refs"])
    if cfg_dropout > 0 and random.random() < cfg_dropout:
        context = empty
    if refs and ref_dropout > 0 and random.random() < ref_dropout:
        refs = []
    return example["latents"], context, refs


def _group_squared_error(
    model: MOSSImageTransformer,
    matcher: FlowMatcher,
    prepared: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Sum squared target residuals for one micro-batch.

    Examples with the same latent and reference shapes share one forward.
    Reference and text tokens are not scored. The caller divides by the
    element count, which for an optimizer step is the total across micro-batches.

    Args:
        model: Training-mode transformer.
        matcher: Sigma sampler and velocity target.
        prepared: Per-example ``(latents, context, refs)``.
        device: Forward device.

    Returns:
        ``(sum of squares, element count)``. The sum is fp32 and keeps its graph.
    """

    if getattr(model, "packed_batching", False):
        return _packed_squared_error(model, matcher, prepared, device)
    groups: dict[tuple, list] = {}
    for latents, context, refs in prepared:
        key = (tuple(latents.shape), tuple(tuple(ref.shape) for ref in refs))
        groups.setdefault(key, []).append((latents, context, refs))
    total_error: torch.Tensor | None = None
    total_count = 0
    for group in groups.values():
        clean = torch.stack([item[0] for item in group]).to(device=device, dtype=torch.float32)
        sigma = matcher.sample_sigma(clean.shape[0], device)
        noise = torch.randn_like(clean)
        noisy, velocity = matcher.noisy_latent(clean, sigma, noise)
        context, mask = _pad_context([item[1] for item in group])
        references = None
        if group[0][2]:
            references = [
                torch.stack([item[2][index] for item in group]).to(device=device, dtype=clean.dtype)
                for index in range(len(group[0][2]))
            ]
        prediction = model(
            noisy.to(dtype=model_dtype(model)),
            matcher.timestep(sigma),
            context.to(device=device, dtype=model_dtype(model)),
            mask.to(device),
            ref_hidden_states=references,
        )
        residual = prediction.float().sub(velocity.float())
        squared = residual.square().sum()
        total_error = squared if total_error is None else total_error + squared
        total_count += residual.numel()
    if total_error is None or total_count < 1:
        raise ValueError("No target tokens to score.")
    return total_error, total_count


def _packed_squared_error(model, matcher, prepared, device):
    """Noise heterogeneous images independently and score one packed traversal."""
    noisy_images, targets, contexts, references, timesteps = [], [], [], [], []
    dtype = model_dtype(model)
    for latent, context, refs in prepared:
        clean = latent.unsqueeze(0).to(device=device, dtype=torch.float32)
        sigma = matcher.sample_sigma(1, device)
        noisy, target = matcher.noisy_latent(clean, sigma, torch.randn_like(clean))
        noisy_images.append(noisy[0].to(dtype))
        targets.append(target[0])
        contexts.append(context.to(device=device, dtype=dtype))
        references.append([reference.to(device=device, dtype=torch.float32) for reference in refs])
        timesteps.append(matcher.timestep(sigma))
    predictions = model.forward_packed(noisy_images, torch.cat(timesteps), contexts,
                                       ref_hidden_states=references)
    if len(predictions) != len(targets):
        raise RuntimeError("Packed model returned the wrong number of target predictions.")
    error = torch.zeros((), device=device)
    count = 0
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape:
            raise RuntimeError("Packed model returned the wrong target shape.")
        error = error + (prediction.float() - target).square().sum()
        count += target.numel()
    return error, count


def _group_loss(
    model: MOSSImageTransformer,
    matcher: FlowMatcher,
    prepared: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]],
    device: torch.device,
) -> torch.Tensor:
    """Mean squared error over the target elements of one batch.

    A larger image weighs more, in proportion to its token count. ``train``
    uses the same reduction across accumulated micro-batches.

    Args:
        model: Training-mode transformer.
        matcher: Sigma sampler and velocity target.
        prepared: Per-example ``(latents, context, refs)``.
        device: Forward device.

    Returns:
        Scalar fp32 loss.
    """

    total_error, total_count = _group_squared_error(model, matcher, prepared, device)
    return total_error / total_count


def _scale_step_error(error: torch.Tensor, total_count: int) -> torch.Tensor:
    """Scale one micro-batch's summed squares by the whole step's element count.

    Equal-sized micro-batches reduce to ``loss / accum``. Differently sized
    ones do not: each element of the step contributes the same gradient.

    Args:
        error: Sum of squared residuals for one micro-batch. Keeps its graph.
        total_count: Target elements across every micro-batch in the step.

    Returns:
        ``error / total_count``.
    """

    if total_count < 1:
        raise ValueError("No target tokens to score.")
    return error / total_count


def _learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warmup followed by cosine decay.

    Args:
        step: Zero-based optimizer step.
        config: Schedule settings.

    Returns:
        Learning rate for this step.
    """

    if step < config.warmup:
        return config.lr * (step + 1) / max(1, config.warmup)
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine)


def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _build_optimizer(parameters: list[torch.nn.Parameter], config: TrainConfig) -> torch.optim.Optimizer:
    from training.optimizers import install_decay_to_one

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay,
                                      betas=tuple(config.adam_betas))
    elif config.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("optimizer adamw8bit requires bitsandbytes.") from exc
        optimizer = bnb.optim.AdamW8bit(parameters, lr=config.lr, weight_decay=config.weight_decay,
                                       betas=tuple(config.adam_betas))
    elif config.optimizer == "adafactor":
        from training.optimizers import build_adafactor

        optimizer = build_adafactor(parameters, config.lr, config.weight_decay)
    else:
        raise ValueError("optimizer must be adamw, adamw8bit or adafactor.")
    install_decay_to_one(optimizer)
    return optimizer


def _write_config(model: MOSSImageTransformer, directory: Path) -> None:
    payload = {
        "_class_name": "MOSSImageTransformer",
        "dim": model.dim,
        "ffn_dim": model.ffn_dim,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "text_dim": model.text_dim,
        "freq_dim": model.freq_dim,
        "latent_channels": model.latent_channels,
        "rope_base": model.rope_base,
    }
    (directory / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    model: MOSSImageTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    directory: Path,
) -> None:
    """Save adapters or the full transformer, plus optimizer state.

    Args:
        model: Model being trained.
        optimizer: Optimizer at this step.
        step: Completed step, zero-based.
        config: Selects LoRA or full weights.
        directory: Checkpoint directory.
    """

    manager = getattr(model, "block_swap_manager", None)
    if manager is not None and hasattr(manager, "commit"):
        manager.commit()
    if directory.exists():
        raise FileExistsError(f"Checkpoint already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", suffix=".partial", dir=directory.parent))
    try:
        _record_config(config, staging)
        if config.mode == "lora":
            save_lora(model, staging / "lora.safetensors")
        else:
            _write_config(model, staging)
            save_transformer(model, staging / "diffusion_pytorch_model.safetensors",
                             comfy_layout=config.checkpoint_format == "comfy")
        torch.save({"step": step, "optimizer": optimizer.state_dict(),
                    "cache_fingerprint": cache_fingerprint(Path(config.cache_dir), config.expected_cache_fingerprint),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   staging / "train_state.pt")
        staging.rename(directory)
    except BaseException:
        if staging.resolve().parent != directory.parent.resolve():
            raise RuntimeError("Checkpoint staging directory escaped its output directory.")
        shutil.rmtree(staging)
        raise


def _require_budget(rows: list[dict[str, object]], resolution: int) -> None:
    """Reject cached images larger than the configured pixel budget.

    Args:
        rows: Index rows. ``height`` and ``width`` are pixels. Each reference
            shape is ``[channels, grid_height, grid_width]``. A ``latent`` entry
            of ``[channels, grid_height, grid_width]`` must match those pixels
            at a stride of 16.
        resolution: Side length of the budget. The area limit is ``resolution ** 2``.
    """

    if resolution < 16:
        raise ValueError("resolution must be at least 16.")
    budget = resolution * resolution
    for row in rows:
        name = str(row.get("file", "example"))
        height, width = int(row["height"]), int(row["width"])
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError(f"{name} is {width}x{height}; sides must be multiples of 16.")
        if height * width > budget:
            raise ValueError(f"{name} is {width}x{height}, above the {resolution}px budget.")
        latent = row.get("latent")
        if isinstance(latent, (list, tuple)) and len(latent) >= 2:
            grid_h, grid_w = int(latent[-2]), int(latent[-1])
            if grid_h * 16 != height or grid_w * 16 != width:
                raise ValueError(
                    f"{name} lists {width}x{height} pixels but a {grid_w}x{grid_h} latent grid."
                )
        for ref in row.get("refs", []):
            ref_h, ref_w = int(ref[-2]), int(ref[-1])
            if ref_h < 1 or ref_w < 1:
                raise ValueError(f"{name} has an empty reference grid.")
            if (ref_h * 16) * (ref_w * 16) > budget:
                raise ValueError(f"{name} has a reference above the {resolution}px budget.")


def _record_config(config: TrainConfig, directory: Path) -> None:
    """Write the resolved training config, including ``resolution``.

    Args:
        config: Run configuration.
        directory: Destination directory. It is created when missing.
    """

    directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    if config.mode == "lora" and config.model:
        payload["model"] = str(Path(config.model).resolve())
    (directory / "train_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_rows(config: TrainConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], torch.Tensor]:
    cache = Path(config.cache_dir)
    cache_fingerprint(cache, config.expected_cache_fingerprint)
    rows = read_index(cache)
    _require_budget(rows, config.resolution)
    holdout = config.n_eval_holdout
    if holdout < 0 or holdout >= len(rows):
        raise ValueError("n_eval_holdout must leave at least one training row.")
    train_rows, eval_rows = (rows, []) if holdout == 0 else (rows[:-holdout], rows[-holdout:])
    empty_path = cache / "empty.safetensors"
    if not empty_path.is_file():
        raise FileNotFoundError(f"Missing unconditional text cache: {empty_path}")
    empty = load_file(str(empty_path))["context"]
    if config.max_tokens:
        from training.token_budget import index_tokens

        for row in rows:
            row["tokens"] = index_tokens(row, cache, empty.shape[0])
            if row["tokens"] > config.max_tokens:
                raise ValueError(f"{row['file']} needs {row['tokens']} tokens, above max_tokens={config.max_tokens}.")
    return train_rows, eval_rows, empty


def _prepare_batch(
    rows: list[dict[str, object]],
    empty: torch.Tensor,
    config: TrainConfig,
    train: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
    """Load one micro-batch and apply training dropout.

    Args:
        rows: Index rows. Training samples ``config.batch`` of them. Evaluation
            scores every row.
        empty: Unconditional text states.
        config: Batch size, cache directory, and dropout probabilities.
        train: ``True`` applies caption and reference dropout.

    Returns:
        ``(latents, context, refs)`` for each chosen example.
    """

    cache = Path(config.cache_dir)
    chosen = rows if not train else random.sample(rows, k=min(config.batch, len(rows)))
    prepared = []
    used_tokens = 0
    for row in chosen:
        example = read_example(cache / str(row["file"]))
        item = (_prepare_example(example, empty, config.cfg_dropout_prob, config.ref_dropout_prob) if train else
                (example["latents"], example["context"], list(example["refs"])))
        if config.max_tokens and train:
            from training.token_budget import prepared_tokens

            count = prepared_tokens(item)
            if count > config.max_tokens:
                raise ValueError(f"Example needs {count} tokens, above max_tokens={config.max_tokens}.")
            if used_tokens + count > config.max_tokens:
                logger.debug(
                    "Skipping example in micro-batch: adding %d tokens exceeds max_tokens=%d (used %d across %d examples).",
                    count, config.max_tokens, used_tokens, len(prepared),
                )
                continue
            used_tokens += count
        prepared.append(item)
    if not prepared:
        raise ValueError(f"No examples could fit within max_tokens={config.max_tokens}.")
    return prepared


def _batch_loss(
    model: MOSSImageTransformer,
    matcher: FlowMatcher,
    rows: list[dict[str, object]],
    empty: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    train: bool,
) -> torch.Tensor:
    """Mean loss of one micro-batch, used for evaluation and a single step."""

    return _group_loss(model, matcher, _prepare_batch(rows, empty, config, train), device)


def _note_resumed_alpha(model: MOSSImageTransformer, config: TrainConfig) -> None:
    """Record the alpha ``load_lora`` installed, when every module agrees.

    The YAML alpha is only the value used to inject a fresh adapter. A resumed
    file replaces it. Writing that value back keeps ``train_config.json``
    honest. Mixed per-module alphas stay as written in the YAML.

    Args:
        model: Transformer after ``load_lora``.
        config: Run configuration updated in place.
    """

    alphas = {float(module.alpha) for module in model.modules() if isinstance(module, LoRALinear)}
    if len(alphas) == 1:
        config.alpha = alphas.pop()
    elif len(alphas) > 1:
        logger.warning(
            "Resumed LoRA modules disagree on alpha %s; train_config.json keeps the YAML value %s.",
            sorted(alphas),
            config.alpha,
        )


def train(config: TrainConfig) -> None:
    """Run LoRA or full fine-tuning.

    Args:
        config: Loaded training config. ``model`` and ``cache_dir`` must exist.
    """

    validate_config(config)
    if not config.model and (config.mode == "lora" or not (config.resume or config.warm_start)):
        raise ValueError("config.model must point at a MOSS-Image bundle or transformer checkpoint.")
    if config.resume:
        state_path = Path(config.resume) / "train_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError("resume requires train_state.pt with optimizer, step and RNG state; use warm_start for weights only.")
        recorded = Path(config.resume) / "train_config.json"
        if recorded.is_file():
            previous = json.loads(recorded.read_text(encoding="utf-8"))
            if previous.get("optimizer") != config.optimizer or previous.get("mode") != config.mode:
                raise ValueError("Resume optimizer/mode differs from the checkpoint; use warm_start to change it.")
            if config.mode == "lora":
                saved_model = previous.get("resolved_model", previous.get("model", ""))
                if Path(saved_model).resolve() != Path(config.model).resolve():
                    raise ValueError("LoRA resume frozen base differs from the checkpoint; use warm_start to change it.")
                if previous.get("base_quantization", "none") != config.base_quantization:
                    raise ValueError("LoRA resume base_quantization differs from the checkpoint; use warm_start to change it.")
                if config.base_quantization == "int8_convrot" and previous.get("convrot_group_size", 256) != config.convrot_group_size:
                    raise ValueError("LoRA resume convrot_group_size differs from the checkpoint; use warm_start to change it.")
                if previous.get("rank") is not None and previous.get("rank") != config.rank:
                    raise ValueError("LoRA resume rank differs from the checkpoint; use warm_start to change it.")
                if previous.get("targets") is not None and previous.get("targets") != config.targets:
                    raise ValueError("LoRA resume targets differ from the checkpoint; use warm_start to change it.")
            elif config.mode == "full":
                if previous.get("full_parameter_dtype") is not None and previous.get("full_parameter_dtype") != config.full_parameter_dtype:
                    raise ValueError("Full resume full_parameter_dtype differs from the checkpoint; use warm_start to change it.")
            defaults = TrainConfig()
            for field in ("adam_betas", "weight_decay", "weight_decay_policy", "qk_norm_decay_to_one"):
                saved = previous.get(field, getattr(defaults, field))
                current = getattr(config, field)
                if field == "adam_betas":
                    saved, current = tuple(saved), tuple(current)
                if saved != current:
                    raise ValueError(f"Resume {field} differs from the checkpoint; use warm_start to change optimizer settings.")
    if config.mixed_precision not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be no or bf16.")
    if config.checkpoint_format not in {"released", "comfy"}:
        raise ValueError("checkpoint_format must be released or comfy.")
    if config.base_quantization not in {"none", "int8", "int8_convrot"}:
        raise ValueError("base_quantization must be none, int8 or int8_convrot.")
    if config.base_quantization != "none" and config.mode != "lora":
        raise ValueError("int8 base weights are frozen and require mode lora.")
    fused = config.optimizer == "adafactor_fused"
    if config.block_swap_mode not in {"sync", "async", "async_full"}:
        raise ValueError("block_swap_mode must be sync, async or async_full.")
    if config.full_parameter_dtype not in {"fp32", "bf16"}:
        raise ValueError("full_parameter_dtype must be fp32 or bf16.")
    if config.full_parameter_dtype == "bf16" and (config.mode != "full" or not fused):
        raise ValueError("BF16 full parameters require full mode and optimizer adafactor_fused.")
    if fused and (config.mode != "full" or config.grad_clip != 0):
        raise ValueError("adafactor_fused requires full mode and grad_clip: 0 (updates occur in backward).")
    if fused and (int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
    )):
        raise ValueError("adafactor_fused supports one process/GPU only.")
    if config.block_swap_mode == "async_full" and (not fused or not config.grad_checkpointing):
        raise ValueError("async_full requires adafactor_fused and grad_checkpointing.")
    if fused and config.blocks_to_swap and config.block_swap_mode != "async_full":
        raise ValueError("Streamed fused full training requires block_swap_mode: async_full.")
    if config.blocks_to_swap and config.block_swap_mode == "async" and config.mode != "lora":
        raise ValueError("async H2D-only block swapping requires mode lora; full training uses sync.")
    if config.blocks_to_swap and config.block_swap_mode == "sync" and config.optimizer not in {"adamw", "adafactor"}:
        raise ValueError("CPU block masters require optimizer adamw or adafactor.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fused and device.type != "cuda":
        raise ValueError("Production adafactor_fused training requires CUDA.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train_rows, eval_rows, empty = _load_rows(config)
    model_path = (config.resume or config.warm_start or config.model) if config.mode == "full" else config.model
    parameter_dtype = torch.bfloat16 if device.type == "cuda" and (
        config.mode == "lora" or config.full_parameter_dtype == "bf16"
    ) else torch.float32
    model = load_transformer(Path(model_path), gradient_checkpointing=config.grad_checkpointing,
                             dtype=parameter_dtype)
    if config.mode == "lora":
        model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        inject_lora(model, config.rank, config.alpha, config.targets)
        if config.resume or config.warm_start:
            adapter = Path(config.resume or config.warm_start)
            load_lora(model, adapter / "lora.safetensors" if adapter.is_dir() else adapter)
            _note_resumed_alpha(model, config)
        if config.base_quantization == "int8":
            from training.quantization import quantize_frozen_linears

            quantize_frozen_linears(model, backend=config.quantized_backend)
        elif config.base_quantization == "int8_convrot":
            from training.convrot import quantize_frozen_convrot_linears

            quantize_frozen_convrot_linears(model, config.convrot_group_size, backend=config.quantized_backend)
        parameters = lora_parameters(model)
    else:
        model.to(dtype=parameter_dtype)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.checkpoint_fraction = config.checkpoint_fraction
    model.activation_offload = config.activation_offload
    model.packed_batching = config.packed_batching
    configure_model_performance(model, config.attention_backend, config.compile_blocks,
                                config.compile_mode, config.lora_compute_dtype)
    place_model(model, device, config.blocks_to_swap, block_swap_mode=config.block_swap_mode)
    model.train()
    from training.optimizers import parameter_groups

    groups = parameter_groups(model.named_parameters(), config.weight_decay, config.weight_decay_policy,
                              qk_norm_decay_to_one=config.qk_norm_decay_to_one, depth=model.num_layers)
    if fused:
        from training.fused_adafactor import FusedAdafactor

        options = {name: {"weight_decay": group["decay_to_one_coef"] if group["decay_to_one"] else group["weight_decay"],
                          "decay_to_one": group["decay_to_one"]} for group in groups for name in group["names"]}
        optimizer = FusedAdafactor(model.named_parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay, state_device=device,
                                  gradient_accumulation_device=config.gradient_accumulation_device,
                                  parameter_options=options)
        optimizer.install_hooks(getattr(model, "block_swap_manager", None))
    else:
        optimizer = _build_optimizer(groups, config)
    start = 0
    if config.resume:
        state_path = Path(config.resume) / "train_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            required = {"step", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
            if not required.issubset(state):
                raise ValueError(f"Resume state is incomplete: missing {sorted(required - set(state))}.")
            current_fingerprint = cache_fingerprint(Path(config.cache_dir), config.expected_cache_fingerprint)
            if state.get("cache_fingerprint", "") != current_fingerprint:
                raise ValueError("Resume cache fingerprint differs from the checkpoint cache.")
            optimizer.load_state_dict(state["optimizer"])
            for parameter, slot in ([] if fused else optimizer.state.items()):
                for key, value in slot.items():
                    if torch.is_tensor(value):
                        slot[key] = value.to(parameter.device) if key != "step" else value.cpu()
            start = int(state["step"]) + 1
            if fused and optimizer.global_step != start:
                raise ValueError("Fused optimizer step does not match the saved training step.")
            if "python_rng" in state:
                random.setstate(state["python_rng"])
                np.random.set_state(state["numpy_rng"])
                torch.set_rng_state(state["torch_rng"])
                if state["cuda_rng"] is not None and device.type == "cuda":
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
    matcher = FlowMatcher(
        config.flow_total, config.flow_mu, config.flow_width, config.flow_cutoff, config.flow_uniform
    )
    output = config.output_path()
    output.mkdir(parents=True, exist_ok=True)
    _record_config(config, output)
    metrics = output / "metrics.jsonl"
    logger.info("training %s from step %s on %s examples", config.mode, start, len(train_rows))
    optimizer.zero_grad(set_to_none=True)
    for step in range(start, config.steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        learning_rate = _learning_rate(step, config)
        _set_lr(optimizer, learning_rate)
        batches = [_prepare_batch(train_rows, empty, config, train=True) for _ in range(config.accum)]
        total_count = sum(example[0].numel() for batch in batches for example in batch)
        total_error = torch.zeros((), device=device)
        work_batches = [[example] for batch in batches for example in batch] if fused and not config.packed_batching else batches
        if fused:
            optimizer.begin_step()
        for batch_index, batch in enumerate(work_batches):
            if fused:
                optimizer.set_step_enabled(batch_index == len(work_batches) - 1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.mixed_precision == "bf16"):
                error, count = _group_squared_error(model, matcher, batch, device)
            expected = sum(example[0].numel() for example in batch)
            if count != expected:
                raise RuntimeError(
                    f"Loss counted {count} elements but the micro-batch has {expected} target elements."
                )
            if not torch.isfinite(error).item():
                raise FloatingPointError(f"Nonfinite training loss at step {step}; optimizer update cancelled.")
            _scale_step_error(error, total_count).backward()
            manager = getattr(model, "block_swap_manager", None)
            if fused and manager is not None:
                manager.finish_backward()
            total_error = total_error + error.detach()
        if fused:
            gradient_norm = optimizer.finish_backward()
        elif config.grad_clip > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
        else:
            gradient_norm = torch.nn.utils.get_total_norm(
                [parameter.grad for parameter in parameters if parameter.grad is not None],
                error_if_nonfinite=True,
            )
        if not fused:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = (total_error / total_count).item()
        completed = step + 1
        checkpoint_due = completed % config.ckpt_every == 0 or completed == config.steps
        evaluation_due = bool(eval_rows) and (completed % (config.eval_every or config.ckpt_every) == 0 or completed == config.steps)
        if step % config.log_every == 0 or completed == config.steps or evaluation_due:
            record = {"step": step, "loss": mean_loss, "lr": learning_rate,
                      "step_seconds": time.perf_counter() - started,
                      "grad_norm": float(gradient_norm)}
            if device.type == "cuda":
                record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            if evaluation_due:
                from training.validation import evaluate_fixed

                heldout = _prepare_batch(eval_rows, empty, config, train=False)
                record.update(evaluate_fixed(model, heldout, config.eval_sigmas, config.eval_seed,
                                             config.flow_total, device, config.mixed_precision, _group_squared_error))
            logger.info("%s", record)
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if checkpoint_due:
            save_checkpoint(model, optimizer, step, config, output / f"step_{step + 1:06d}")
    if fused:
        optimizer.close()
    manager = getattr(model, "block_swap_manager", None)
    if manager is not None:
        manager.close()


def smoke() -> None:
    """Train a tiny random DiT for two LoRA steps and one full step.

    This checks the edit forward, the flow target, and that a LoRA step moves ``B``.
    It does not load MOSS-Image weights.
    """

    torch.manual_seed(0)
    device = torch.device("cpu")
    model = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=2, text_dim=16,
        freq_dim=8, latent_channels=4, gradient_checkpointing=True,
    ).to(device)
    inject_lora(model, rank=2, alpha=2.0)
    matcher = FlowMatcher(total=32, mu=0.0, width=1.0, cutoff=0.95, uniform_mixture=0.01)
    clean = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 3, 16)
    reference = torch.randn(1, 4, 2, 2)
    before = torch.cat([module.lora_B.detach().clone() for module in model.modules() if hasattr(module, "lora_B")])
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    model.train()
    last = 0.0
    for _ in range(2):
        sigma = matcher.sample_sigma(1, device)
        noise = torch.randn_like(clean)
        noisy, velocity = matcher.noisy_latent(clean, sigma, noise)
        prediction = model(noisy, matcher.timestep(sigma), context, ref_hidden_states=[reference])
        loss = F.mse_loss(prediction, velocity)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last = float(loss.detach())
    if not math.isfinite(last):
        raise RuntimeError("Smoke LoRA loss was not finite.")
    after = torch.cat([module.lora_B.detach().clone() for module in model.modules() if hasattr(module, "lora_B")])
    if torch.equal(before, after):
        raise RuntimeError("Smoke LoRA step did not change the adapter.")
    full = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=1, text_dim=16, freq_dim=8, latent_channels=4,
    )
    sigma = matcher.sample_sigma(1, device)
    noise = torch.randn_like(clean)
    noisy, velocity = matcher.noisy_latent(clean, sigma, noise)
    loss = F.mse_loss(full(noisy, matcher.timestep(sigma), context), velocity)
    loss.backward()
    value = float(loss.detach())
    if full.video_embedder.weight.grad is None or not math.isfinite(value):
        raise RuntimeError("Smoke full-finetune backward did not reach the embedder.")
    logger.info("smoke ok, lora loss %.4f", last)


def main(argv: list[str] | None = None) -> None:
    """Command line for LoRA and full fine-tuning."""

    parser = argparse.ArgumentParser(description="Train a MOSS-Image LoRA or full fine-tune.")
    parser.add_argument("--config", type=Path, help="YAML config. Not used with --smoke.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU training check and exit.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.smoke:
        smoke()
        return
    if args.config is None:
        raise SystemExit("Provide --config or --smoke.")
    train(load_config(args.config))


if __name__ == "__main__":
    main()
