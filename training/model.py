"""MOSS-Image DiT with the released state-dict names and an optional reference axis.

Text-to-image behavior follows the forward in the OpenMOSS pipeline
(``pipeline_moss_image.py`` on the MOSS-Image-Base bundle): image tokens, then the
unpadded text tokens, 4D RoPE ``(r, t, h, w)``, shared timestep modulation, and a
velocity head on the image tokens only. With no reference latents every token keeps
``r = 0``, which is the pretrained model.

Edit mode appends clean reference token grids after the target grid and assigns them
``r = 1, 2, ...``. The head still predicts only the target grid. The technical report
keeps this axis so reference-conditioned training does not require a new positional
encoding.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.activation_offload import activation_cpu_offload
from training.performance import model_performance_context, scaled_attention, scaled_attention_varlen


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SiTU-GLU product used by every feed-forward layer.

    Args:
        hidden: Up-projection, arbitrary shape, same shape as ``gate``.
        gate: Gate projection. The cap on this branch is 4.

    Returns:
        Elementwise SiTU-GLU activation, same shape as the inputs.
    """

    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate_rope(tokens: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply the released half-rotation RoPE.

    Args:
        tokens: ``[..., head_dim]`` queries or keys.
        rope: ``[1, seq, head_dim]`` with sine in the first half and cosine in the second.

    Returns:
        Rotated tokens, same shape as ``tokens``.
    """

    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    first, second = tokens.chunk(2, -1)
    return tokens * cos + torch.cat((-second, first), -1) * sin


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rope: torch.Tensor,
    heads: int,
    backend: str = "auto",
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint self-attention with RoPE on query and key.

    Args:
        query: ``[batch, seq, dim]``.
        key: ``[batch, seq, dim]``.
        value: ``[batch, seq, dim]``.
        rope: ``[1, seq, head_dim]``.
        heads: Number of attention heads. ``dim`` must be divisible by ``heads``.
        backend: Performance backend passed to ``scaled_attention``.
        cu_seqlens: Packed sequence boundaries, or ``None`` for ordinary attention.

    Returns:
        Attention output ``[batch, seq, dim]``.
    """

    shape = (*query.shape[:2], heads, query.shape[-1] // heads)
    query = rotate_rope(query.reshape(shape), rope).transpose(1, 2)
    key = rotate_rope(key.reshape(shape), rope).transpose(1, 2)
    value = value.reshape(shape).transpose(1, 2)
    if cu_seqlens is None:
        mixed = scaled_attention(query, key, value, backend)
    else:
        mixed = scaled_attention_varlen(query, key, value, cu_seqlens, backend)
    return mixed.transpose(1, 2).flatten(2)


def rope_coordinates(images: list[tuple[int, int, int]], text_length: int, device: torch.device) -> torch.Tensor:
    """Build ``(r, t, h, w)`` coordinates for image grids followed by text.

    Args:
        images: ``(reference_index, height, width)`` in token-grid units, target first.
            ``reference_index`` is 0 for the target. Height and width are at least 1.
        text_length: Number of real text tokens. Must be at least 1.
        device: Device of the returned tensor.

    Returns:
        Float tensor ``[image_tokens + text_length, 4]``. Image ``t`` is ``text_length``
        and spatial axes are centered. Text uses ``r = 0``, ``t = 0..text_length-1``,
        and zero spatial coordinates.
    """

    if text_length < 1:
        raise ValueError("text_length must be positive.")
    if not images:
        raise ValueError("At least the target image grid is required.")
    parts: list[torch.Tensor] = []
    for reference_index, height, width in images:
        if height < 1 or width < 1:
            raise ValueError("Image grids must be non-empty.")
        rows, cols = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = torch.stack(
            (
                torch.full_like(rows, reference_index),
                torch.full_like(rows, text_length),
                rows,
                cols,
            ),
            -1,
        ).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        parts.append(image)
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    return torch.cat((*parts, text), 0)


def rope_phases(coordinates: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
    """Convert 4D coordinates into the sine/cosine tensor consumed by ``rotate_rope``.

    Args:
        coordinates: ``[seq, 4]`` float coordinates.
        bands: ``[num_bands]`` fp32 rotary frequencies. ``8 * num_bands`` is the head dim.

    Returns:
        ``[1, seq, 8 * num_bands]`` with all four sines packed before all four cosines.
    """

    phase = coordinates.unsqueeze(-1) * bands.to(device=coordinates.device, dtype=torch.float32)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class RMSNorm(nn.Module):
    """RMSNorm with a learned scale, computed in fp32 and cast back to the input dtype."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension.

        Args:
            tokens: ``[..., dim]``.

        Returns:
            Normalized tokens, same shape and dtype as ``tokens``.
        """

        normalized = tokens.float() * torch.rsqrt(tokens.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(tokens.dtype) * self.weight


class TextBlock(nn.Module):
    """Two-layer text adapter block. It refines text tokens before they join the image stream."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_backend = "auto"
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine text tokens.

        Args:
            tokens: ``[batch, text_length, dim]``.
            rope: Text slice of the RoPE tensor, ``[1, text_length, head_dim]``.

        Returns:
            ``[batch, text_length, dim]``.
        """

        query, key, value = self.qkv(self.norm1(tokens)).chunk(3, -1)
        query = self.q_norm(query.unflatten(-1, (self.heads, -1))).flatten(2)
        key = self.k_norm(key.unflatten(-1, (self.heads, -1))).flatten(2)
        tokens = tokens + self.proj(attention(
            query, key, value, rope, self.heads, self.attention_backend, cu_seqlens,
        ))
        hidden = self.norm2(tokens)
        return tokens + self.ffn_down(situ(self.ffn_up(hidden), self.ffn_gate(hidden)))


class TextProjection(nn.Module):
    """Project frozen text-encoder states into the DiT width and run two text blocks."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project and refine text tokens.

        Args:
            tokens: ``[batch, text_length, text_dim]`` hidden states.
            rope: Text RoPE, ``[1, text_length, head_dim]``.

        Returns:
            ``[batch, text_length, dim]``.
        """

        hidden = self.proj(tokens.to(_linear_dtype(self.proj)))
        for block in self.blocks:
            hidden = block(hidden, rope, cu_seqlens)
        return hidden


class Block(nn.Module):
    """One joint image/text transformer block with shared scale-shift-gate modulation."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_backend = "auto"
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to the eager or shared compiled block core."""

        return model_performance_context(self).call_block(self, tokens, rope, modulation, cu_seqlens)

    def _forward_impl(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention and the gated feed-forward on one sample.

        Args:
            tokens: ``[1, seq, dim]`` target, reference, and text tokens.
            rope: ``[1, seq, head_dim]``.
            modulation: ``[1, 6, dim]`` shared across the sequence.

        Returns:
            ``[1, seq, dim]``.
        """

        if modulation.ndim != 3 or modulation.shape[1] != 6:
            raise ValueError("modulation must be [samples, 6, dim].")
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.unbind(1)
        sample_indices = _modulation_sample_indices(modulation, tokens, cu_seqlens)
        hidden = self.norm1(tokens)
        hidden = hidden * (1 + _condition_for_tokens(scale_attn, sample_indices))
        hidden = hidden + _condition_for_tokens(shift_attn, sample_indices)
        weight_dtype = _linear_dtype(self.proj["q"])
        hidden = hidden.to(weight_dtype)
        query = self.q_norm(self.proj["q"](hidden).unflatten(-1, (self.heads, -1))).flatten(2)
        key = self.k_norm(self.proj["k"](hidden).unflatten(-1, (self.heads, -1))).flatten(2)
        attended = attention(
            query,
            key,
            self.proj["v"](hidden),
            rope,
            self.heads,
            self.attention_backend,
            cu_seqlens,
        )
        tokens = tokens + self.proj["o"](attended) * _condition_for_tokens(gate_attn, sample_indices)
        hidden = self.norm2(tokens)
        hidden = hidden * (1 + _condition_for_tokens(scale_ffn, sample_indices))
        hidden = hidden + _condition_for_tokens(shift_ffn, sample_indices)
        up, gate = self.proj["ffn_up_gate"](hidden.to(weight_dtype)).chunk(2, -1)
        return tokens + self.proj["ffn_down"](situ(up, gate)) * _condition_for_tokens(
            gate_ffn, sample_indices,
        )


class RoPE(nn.Module):
    """4D RoPE. Frequencies are a fp32 function of ``rope_base``, not learned weights."""

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self.default_bands(torch.device("cpu")))

    def default_bands(self, device: torch.device) -> torch.Tensor:
        """Return the training-time CPU fp32 frequencies moved onto ``device``.

        Args:
            device: Destination device.

        Returns:
            ``[num_bands]`` fp32 frequencies.
        """

        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse: bool = True):
        returned = super()._apply(fn, recurse=recurse)
        self.bands = self.default_bands(self.bands.device)
        return returned

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self.default_bands(self.bands.device)

    def forward(self, images: list[tuple[int, int, int]], text_length: int, device: torch.device) -> torch.Tensor:
        """Build RoPE for the given grids.

        Args:
            images: ``(reference_index, height, width)`` token grids.
            text_length: Real text tokens.
            device: Activation device.

        Returns:
            ``[1, seq, head_dim]`` in the video activation dtype is applied by the caller.
            This method returns fp32.
        """

        coordinates = rope_coordinates(images, text_length, device)
        return rope_phases(coordinates, self.bands)


class MOSSImageTransformer(nn.Module):
    """5B-class single-stream DiT. Defaults match MOSS-Image-Base.

    Args:
        dim: Joint hidden size.
        ffn_dim: Feed-forward intermediate size.
        num_heads: Attention heads. ``dim // num_heads`` must be divisible by 8.
        num_layers: Joint blocks, excluding the two text-adapter blocks.
        text_dim: Frozen text-encoder width.
        freq_dim: Sinusoidal timestep width. Must be even.
        latent_channels: Patched VAE channels, 128 for FLUX.2 2x2 patches.
        rope_base: Rotary base.
        gradient_checkpointing: Recompute joint blocks during training.
        checkpoint_fraction: Deterministic fraction of resident joint blocks to checkpoint.
        activation_offload: Move autograd-saved CUDA activations to pinned CPU memory.
        packed_batching: Records whether the trainer selected packed batching.
    """

    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
        gradient_checkpointing: bool = False,
        checkpoint_fraction: float = 1.0,
        activation_offload: bool = False,
        packed_batching: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.rope_base = rope_base
        self.gradient_checkpointing = gradient_checkpointing
        _checkpointed_layer_indices(num_layers, checkpoint_fraction)
        self.checkpoint_fraction = float(checkpoint_fraction)
        self.activation_offload = bool(activation_offload)
        self.packed_batching = bool(packed_batching)
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The computational activation/base weight dtype of the transformer."""
        embedder = getattr(self, "video_embedder", None)
        if embedder is not None:
            if hasattr(embedder, "compute_dtype"):
                return embedder.compute_dtype
            if hasattr(embedder, "weight") and embedder.weight is not None:
                return embedder.weight.dtype
        if hasattr(self, "layers") and len(self.layers) > 0:
            layer = self.layers[0]
            if hasattr(layer, "norm1") and hasattr(layer.norm1, "weight"):
                return layer.norm1.weight.dtype
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        skip_layers: list[int] | tuple[int, ...] | None = None,
        ref_hidden_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict the target-grid velocity.

        Args:
            hidden_states: Noisy target latents ``[batch, latent_channels, height, width]``.
            timestep: Diffusion time on the training grid, typically ``1000 * sigma``.
                A scalar broadcasts over the batch. A 1D tensor must have length ``batch``.
            encoder_hidden_states: Text states ``[batch, seq, text_dim]``.
            encoder_attention_mask: Binary ``[batch, seq]`` mask. ``None`` keeps every token.
            skip_layers: Joint-block indices to leave as the identity. Used for the
                negative-branch skip at layer 22. Empty during training.
            ref_hidden_states: Clean reference latents, each ``[batch, latent_channels, hr, wr]``.
                ``None`` or an empty list is text-to-image.

        Returns:
            Velocity ``[batch, latent_channels, height, width]`` for the target grid only.
        """

        skipped = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [batch, latent_channels, height, width].")
        batch, _, height, width = hidden_states.shape
        if (batch > 1 and torch.is_grad_enabled()
                and getattr(self, "block_swap_mode", "sync") == "async_full"
                and getattr(self, "blocks_to_swap", 0)):
            raise ValueError("Trainable block swapping requires sequential singleton backward graphs.")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [batch, seq, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [batch, seq] mask.")
        references = list(ref_hidden_states or [])
        for reference in references:
            if reference.ndim != 4 or reference.shape[0] != batch or reference.shape[1] != self.latent_channels:
                raise ValueError("Each reference must be [batch, latent_channels, height, width].")
        outputs: list[torch.Tensor] = []
        with activation_cpu_offload(self, self.activation_offload, hidden_states.device):
            modulation, head = self._time_conditions(timestep, batch, hidden_states.device)
            for index in range(batch):
                context = encoder_hidden_states[index : index + 1, mask[index].bool()]
                if context.shape[1] == 0:
                    raise ValueError("Each sample needs at least one valid text token.")
                sample_refs = [reference[index : index + 1] for reference in references]
                outputs.append(self._forward_sample(
                    hidden_states[index : index + 1],
                    sample_refs,
                    context,
                    modulation[index : index + 1],
                    head[index : index + 1],
                    skipped,
                ))
        return torch.cat(outputs)

    def forward_packed(
        self,
        hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        ref_hidden_states: list[list[torch.Tensor]] | None = None,
    ) -> list[torch.Tensor]:
        """Predict a heterogeneous batch in one packed transformer traversal.

        Each target is ``[channels, height, width]`` and each context is
        ``[sequence, text_dim]``. References are grouped per example with the
        same unbatched image layout. Attention is isolated by per-example
        cumulative sequence lengths.
        """

        if not hidden_states:
            raise ValueError("forward_packed requires at least one target.")
        batch = len(hidden_states)
        if len(encoder_hidden_states) != batch:
            raise ValueError("encoder_hidden_states must contain one tensor per target.")
        references = ref_hidden_states if ref_hidden_states is not None else [[] for _ in range(batch)]
        if len(references) != batch:
            raise ValueError("ref_hidden_states must contain one reference list per target.")
        device = hidden_states[0].device
        for target in hidden_states:
            if target.ndim != 3 or target.shape[0] != self.latent_channels:
                raise ValueError("Each packed target must be [latent_channels, height, width].")
            if target.device != device:
                raise ValueError("All packed targets, contexts, and references must share one device.")
        for context in encoder_hidden_states:
            if context.ndim != 2 or context.shape[0] == 0 or context.shape[1] != self.text_dim:
                raise ValueError("Each packed context must be non-empty [sequence, text_dim].")
            if context.device != device:
                raise ValueError("All packed targets, contexts, and references must share one device.")
        for sample_refs in references:
            for reference in sample_refs:
                if reference.ndim != 3 or reference.shape[0] != self.latent_channels:
                    raise ValueError("Each packed reference must be [latent_channels, height, width].")
                if reference.device != device:
                    raise ValueError("All packed targets, contexts, and references must share one device.")
        with activation_cpu_offload(self, self.activation_offload, device):
            modulation, head = self._time_conditions(timestep, batch, device)
            return self._forward_packed_impl(hidden_states, encoder_hidden_states, references, modulation, head)

    def _time_conditions(
        self,
        timestep: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
        if times.numel() not in {1, batch}:
            raise ValueError("timestep must be scalar or contain one value per sample.")
        times = times.expand(batch)
        half = self.freq_dim // 2
        frequencies = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * frequencies
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(_linear_dtype(self.time_embedding[0]))).float()
        modulation = self.shared_block_modulation(
            time.to(_linear_dtype(self.shared_block_modulation)),
        ).view(batch, 6, -1)
        head = self.video_head_modulation(
            time.to(_linear_dtype(self.video_head_modulation)),
        ).view(batch, 2, -1)
        return modulation, head

    def _forward_packed_impl(
        self,
        hidden_states: list[torch.Tensor],
        contexts: list[torch.Tensor],
        references: list[list[torch.Tensor]],
        modulation: torch.Tensor,
        head: torch.Tensor,
    ) -> list[torch.Tensor]:
        videos: list[torch.Tensor] = []
        ropes: list[torch.Tensor] = []
        text_ropes: list[torch.Tensor] = []
        text_lengths: list[int] = []
        target_shapes: list[tuple[int, int]] = []
        target_lengths: list[int] = []
        joint_lengths: list[int] = []
        for target, context, sample_refs in zip(hidden_states, contexts, references):
            height, width = target.shape[-2:]
            target_shapes.append((height, width))
            target_lengths.append(height * width)
            grids = [(0, height, width)]
            pieces = [_project_video(self.video_embedder, target.unsqueeze(0))]
            for reference_index, reference in enumerate(sample_refs, start=1):
                ref_height, ref_width = reference.shape[-2:]
                grids.append((reference_index, ref_height, ref_width))
                pieces.append(_project_video(self.video_embedder, reference.unsqueeze(0)))
            video = torch.cat(pieces, 1)
            sample_rope = self.rope(grids, context.shape[0], target.device).to(video.dtype)
            videos.append(video)
            ropes.append(sample_rope)
            text_ropes.append(sample_rope[:, video.shape[1] :])
            text_lengths.append(context.shape[0])
            joint_lengths.append(video.shape[1] + context.shape[0])

        text_cu = _cumulative_lengths(text_lengths, hidden_states[0].device)
        packed_text = torch.cat([context.unsqueeze(0) for context in contexts], 1)
        packed_text_rope = torch.cat(text_ropes, 1)
        packed_text = self.text_embedder(packed_text, packed_text_rope, text_cu)
        text_parts = list(packed_text.split(text_lengths, dim=1))
        token_parts = [torch.cat((video, text), 1) for video, text in zip(videos, text_parts)]
        tokens = torch.cat(token_parts, 1)
        rope = torch.cat(ropes, 1)
        if self.training and not tokens.requires_grad:
            tokens.requires_grad_(True)
        joint_cu = _cumulative_lengths(joint_lengths, hidden_states[0].device)
        tokens = self._run_layers(tokens, rope, modulation, frozenset(), joint_cu)

        outputs: list[torch.Tensor] = []
        offset = 0
        for index, ((height, width), target_length, joint_length) in enumerate(
            zip(target_shapes, target_lengths, joint_lengths),
        ):
            predicted = tokens[:, offset : offset + target_length]
            predicted = self.final_norm_video(predicted.to(self.final_norm_video.weight.dtype))
            predicted = predicted * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            predicted = self.final_linear_video(predicted.to(_linear_dtype(self.final_linear_video)))
            outputs.append(predicted.transpose(1, 2).reshape(-1, height, width))
            offset += joint_length
        return outputs

    def _forward_sample(
        self,
        hidden_states: torch.Tensor,
        references: list[torch.Tensor],
        context: torch.Tensor,
        modulation: torch.Tensor,
        head: torch.Tensor,
        skipped: frozenset[int],
    ) -> torch.Tensor:
        """Forward one sample.

        Args:
            hidden_states: ``[1, channels, height, width]`` target.
            references: Clean reference latents for this sample.
            context: ``[1, text_length, text_dim]`` with padding already removed.
            modulation: ``[1, 6, dim]``.
            head: ``[1, 2, dim]`` output scale and shift.
            skipped: Block indices that stay at the identity.

        Returns:
            ``[1, channels, height, width]`` target velocity.
        """

        _, _, height, width = hidden_states.shape
        grids = [(0, height, width)]
        videos = [_project_video(self.video_embedder, hidden_states)]
        for reference_index, reference in enumerate(references, start=1):
            _, _, ref_height, ref_width = reference.shape
            grids.append((reference_index, ref_height, ref_width))
            videos.append(_project_video(self.video_embedder, reference))
        video = torch.cat(videos, 1)
        target_tokens = height * width
        rope = self.rope(grids, context.shape[1], hidden_states.device).to(video.dtype)
        # Text RoPE is the tail. Target and reference tokens occupy the front.
        text = self.text_embedder(context, rope[:, video.shape[1] :])
        tokens = torch.cat((video, text), 1)
        if self.training and not tokens.requires_grad:
            tokens.requires_grad_(True)
        tokens = self._run_layers(tokens, rope, modulation, skipped)
        predicted = self.final_norm_video(tokens[:, :target_tokens].to(self.final_norm_video.weight.dtype))
        predicted = predicted * (1 + head[:, 1, None]) + head[:, 0, None]
        predicted = self.final_linear_video(predicted.to(_linear_dtype(self.final_linear_video)))
        return predicted.transpose(1, 2).reshape(1, -1, height, width)

    def _run_layers(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        skipped: frozenset[int],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run joint blocks with swapping and selective checkpointing."""

        swap_boundary = len(self.layers) - getattr(self, "blocks_to_swap", 0)
        block_swap_manager = getattr(self, "block_swap_manager", None)
        streamed = [index for index in range(swap_boundary, len(self.layers)) if index not in skipped]
        streamed_position = {index: position for position, index in enumerate(streamed)}
        checkpointed = _checkpointed_layer_indices(len(self.layers), self.checkpoint_fraction)
        for layer_index, layer in enumerate(self.layers):
            if layer_index in skipped:
                continue
            inputs = (tokens, rope, modulation) if cu_seqlens is None else (tokens, rope, modulation, cu_seqlens)
            if block_swap_manager is not None and layer_index >= swap_boundary:
                position = streamed_position[layer_index]
                forward_prefetch = streamed[position + 1] if position + 1 < len(streamed) else None
                backward_prefetch = streamed[position - 1] if position else None
                tokens = block_swap_manager.block(
                    layer_index,
                    *inputs,
                    forward_prefetch=forward_prefetch,
                    backward_prefetch=backward_prefetch,
                    checkpoint_block=torch.is_grad_enabled(),
                )
            elif layer_index >= swap_boundary:
                from training.memory import offloaded_block

                tokens = offloaded_block(layer, *inputs)
            elif self.gradient_checkpointing and self.training and layer_index in checkpointed:
                tokens = checkpoint(layer, *inputs, use_reentrant=False)
            else:
                tokens = layer(*inputs)
        return tokens


def _project_video(embedder: nn.Linear, latents: torch.Tensor) -> torch.Tensor:
    """Flatten a latent grid and project it to the DiT width.

    Args:
        embedder: ``video_embedder``.
        latents: ``[1, channels, height, width]``.

    Returns:
        ``[1, height * width, dim]``.
    """

    tokens = latents.flatten(2).transpose(1, 2)
    return embedder(tokens.to(_linear_dtype(embedder)).contiguous())


def model_dtype(model: nn.Module) -> torch.dtype:
    """Return the computational activation/base weight dtype of the transformer."""
    if hasattr(model, "compute_dtype"):
        return model.compute_dtype
    embedder = getattr(model, "video_embedder", None)
    if embedder is not None:
        if hasattr(embedder, "compute_dtype"):
            return embedder.compute_dtype
        if hasattr(embedder, "weight") and embedder.weight is not None:
            return embedder.weight.dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _linear_dtype(module: nn.Module) -> torch.dtype:
    if hasattr(module, "compute_dtype"):
        return module.compute_dtype
    return module.weight.dtype


def _modulation_sample_indices(
    modulation: torch.Tensor,
    tokens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor | None:
    """Map packed token positions to their per-sample modulation row."""

    if modulation.shape[0] == tokens.shape[0]:
        return None
    if tokens.shape[0] != 1 or cu_seqlens is None or modulation.shape[0] != cu_seqlens.numel() - 1:
        raise ValueError("Packed modulation must contain one row per sequence.")
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
    return torch.repeat_interleave(
        torch.arange(modulation.shape[0], device=modulation.device),
        lengths,
        output_size=tokens.shape[1],
    )


def _condition_for_tokens(condition: torch.Tensor, sample_indices: torch.Tensor | None) -> torch.Tensor:
    if sample_indices is None:
        return condition[:, None]
    return condition.index_select(0, sample_indices).unsqueeze(0)


def _cumulative_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = torch.tensor([0, *lengths], device=device, dtype=torch.int64)
    return values.cumsum(0)


def _checkpointed_layer_indices(depth: int, fraction: float) -> frozenset[int]:
    """Choose the nearest-count, evenly spread deterministic subset of joint blocks."""

    value = float(fraction)
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError("checkpoint_fraction must be between 0 and 1.")
    count = min(depth, int(depth * value + 0.5))
    if count == 0:
        return frozenset()
    return frozenset(((2 * index + 1) * depth) // (2 * count) for index in range(count))


def _validate_skip_layers(layers: list[int] | tuple[int, ...] | None, depth: int) -> frozenset[int]:
    """Return a set of in-range block indices.

    Args:
        layers: Indices to skip, or ``None``.
        depth: Number of joint blocks.

    Returns:
        Valid indices.
    """

    chosen = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in chosen):
        raise ValueError(f"skip_layers must contain integer block indices in [0, {depth}).")
    return frozenset(chosen)
