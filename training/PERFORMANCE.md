# Training performance

The fast paths are opt-in. Focused CPU and CUDA tests cover the numerical and
checkpoint contracts below. Throughput and larger workloads require separate
measurements. The same paths support text-to-image and reference-conditioned examples.

## LoRA

Use `attention_backend: flash_attn_2` with an ABI-compatible FlashAttention 2
installation. `auto` keeps PyTorch SDPA. `sdpa_cudnn` explicitly requests cuDNN
and fails when that kernel cannot support the workload.

Under autocast, LoRA casts its small A/B matrices to the compute dtype rather
than converting the entire token activation to FP32. Adapter master parameters
remain FP32. `lora_compute_dtype` can override the compute dtype explicitly.

`compile_blocks: true` compiles one shared block implementation. Checkpoint and
swap orchestration stay outside that function; checkpoint names stay unchanged.
Torch 2.9 requires Triton 3.5 (`triton-windows` on Windows), a working C++ build
toolchain, and warmup. Compilation startup is not steady-state training speed.
`reduce-overhead` is rejected because its CUDA graphs can retain block copies.

Frozen LoRA bases can use `block_swap_mode: async` and `blocks_to_swap: N`.
Two packed device slots overlap pinned-memory H2D copies with computation.
Use zero swapped blocks when the resident model fits: PCIe transfers cost time.

## Full training

FP32-master mode supports native `adafactor` with synchronous swapping.
Its optimizer uses parameter-relative learning rates.

Fused updates run during backward:

```yaml
mode: full
full_parameter_dtype: bf16
mixed_precision: bf16
optimizer: adafactor_fused
grad_clip: 0
grad_checkpointing: true
block_swap_mode: async_full
blocks_to_swap: 0
attention_backend: flash_attn_2
```

Increase swapped blocks only as needed to fit VRAM. This mode updates each
parameter in its post-accumulate backward hook and releases its gradient.
Factored optimizer moments remain FP32; BF16 weights use stochastic rounding.
`async_full` overlaps packed H2D prefetch and D2H writeback of updated weights.
Checkpoint export drains writeback first. FP32 parameter storage is also
supported with `full_parameter_dtype: fp32`, at greater memory/transfer cost.

This is a different optimizer/precision tradeoff: `adafactor_fused` uses an
absolute manual LR, with no relative parameter scaling. Retune LR; do not copy
native Adafactor LR blindly. Global gradient clipping is incompatible with
updates during backward. A late nonfinite gradient can follow partial updates;
the run fails and requires reloading a checkpoint. Failed steps cannot be saved.

Logical batches and accumulation are processed as sequential singleton
forward/backward graphs with one target-element-weighted optimizer step.
With `packed_batching: true`, a logical microbatch instead uses one segmented
packed graph. The target-element denominator still spans all accumulation
microbatches. Use one process/GPU.

`gradient_accumulation_device: cpu` moves early microbatch gradients to pinned
FP32 host buffers and clears their device storage. The final backward combines
them with the current gradient and updates each parameter once. Parameters used
only in earlier microbatches are updated too. Transfers are synchronous for
correct storage lifetime; this option reduces device gradient memory at a PCIe
cost. The default `device` keeps gradients locally until the final backward.

## Batching and memory controls

| Setting | Behavior |
| --- | --- |
| `checkpoint_fraction: 1.0` | Checkpoint every resident joint block. Values in `[0,1]` select a deterministic subset; swapped blocks always recompute. `grad_checkpointing: false` disables resident checkpointing. |
| `activation_offload: false` | When enabled, offload saved CUDA activations to pinned CPU memory and restore their original layout for backward. Parameters, quantized buffers and staging rings remain in their managed storage. |
| `packed_batching: false` | When enabled, concatenate heterogeneous targets, references and text into one transformer traversal with isolated per-example attention. |
| `max_tokens: 0` | A positive value caps target + reference + text tokens in each microbatch. Oversized individual rows fail before loading the model. Batch size is an upper bound when the token cap is active. |

Packed `flash_attn_2` uses FlashAttention's variable-length kernel. Packed
`auto`/`sdpa` uses a portable block-diagonal boolean mask, which occupies memory
quadratic in the total packed sequence length. Explicit `sdpa_cudnn` does not
support this packed mask and fails clearly. Select FA2 for large packed batches.
The standard per-example path remains available for comparison.

For resident LoRA, measure lower checkpoint fractions before introducing CPU
transfers. For full training with accumulation, compare host gradient storage
against the device default. Activation offload can permit less recomputation or
fewer swapped weights, but transfer time can outweigh those savings.

## Quantized execution

`base_quantization: int8` or `int8_convrot` supports `quantized_backend`:

- `eager` reconstructs floating weights and is the default.
- `triton` uses tiled INT8 weight decoding inside forward/input-gradient GEMMs.
  ConvRot runs normalized Hadamard stages over bounded row workspaces. It does
  not construct a full floating weight matrix or a full rotated activation copy.
  Unsupported dtype/device/layout combinations fail explicitly.
- `auto` selects Triton for supported contiguous CUDA FP16/BF16/FP32 tensors and
  otherwise uses eager execution. Triton is an optional dependency.

Quantized weights remain frozen. Gradients propagate through inputs and LoRA
adapters. Kernel execution is independent of checkpoint and block-swap choices.

## Optimizer and adapter controls

`adam_betas: [0.9, 0.999]` configures AdamW/AdamW8bit. `weight_decay_policy: all`
applies decay to all parameters. `matrix` excludes biases, norms, vectors and
input embedding projections. `qk_norm_decay_to_one: true` separately decays the
last joint block's trainable Q/K norm toward one. Both native and fused optimizers
support the selected decay roles. Changing optimizer settings on a strict resume
requires an explicit warm start.

LoRA `targets: joint` selects only joint-block linears. `joint_and_text` also
selects the text projection/refiner. Legacy `blocks` means joint plus text, and
`blocks_and_io` additionally selects image input/output projections. Exported
adapters record the exact preset so sampling restores the same target set.

## Validation, cache identity and resume

With `n_eval_holdout > 0`, validation runs every `eval_every` completed updates;
zero follows `ckpt_every`. The final update is evaluated as well, independently
of logging cadence. `eval_sigmas: [0.25, 0.5, 0.75]` and `eval_seed: 0` select
repeatable noise levels and a separate noise stream. Metrics contain each
`eval_loss@SIGMA` and their mean `eval_loss`. Evaluation preserves training RNG
and model mode, and scores rows individually to bound activation memory.

Precache applies EXIF orientation before choosing dimensions and encoding pixels.
It records SHA256 identities of source images, model/tokenizer assets (including
chat templates), preprocessing/dtype settings and library versions in
`cache_info.json`. This adds an initial file-hashing
pass. `expected_cache_fingerprint` optionally pins an exact manifest identity.
Older caches without provenance remain usable when no expected fingerprint is
specified. Resuming a checkpoint requires its recorded cache identity to match.

Checkpoints are written into a temporary sibling directory and published by
rename after weights, config, optimizer and RNG state are complete. Existing
checkpoint directories are protected. Full safetensors exports stream tensors
one at a time; fused Comfy QKV bytes are written from separate Q/K/V pieces,
avoiding a second complete host copy or a fused GPU allocation.

`resume` requires optimizer, step and RNG state and continues the saved run.
`warm_start` initializes full-model weights or an adapter with a fresh optimizer
and schedule. For LoRA, `model` still identifies the frozen base and
`warm_start` identifies the adapter file/directory. Choose a fresh output
directory for a warm start. The two options are mutually exclusive.

## Comparable measurements

Report GPU, target/reference pixel dimensions and token counts, batch,
accumulation, rank, precision, checkpointing, swapped blocks, optimizer,
attention backend, compile warmup, and peak allocated VRAM. Measure T2I and edit
separately. One equally sized reference roughly doubles the joint sequence;
attention cost grows quadratically with its length.

## Validation scope

CPU and CUDA checks cover packed attention, saved activation layouts,
host-gradient accumulation, quantized kernels, and LoRA/full checkpoint resume.
Released-model execution checks cover packed LoRA and full-training updates.
These checks verify execution and checkpoint integrity; they do not establish
convergence, image quality, or steady-state throughput. Measure checkpoint I/O
separately from training updates.
