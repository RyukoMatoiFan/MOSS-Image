# Training capabilities and validation

See [performance settings](PERFORMANCE.md) for configuration and tradeoffs.

## Memory and batching

- Selective activation checkpointing and optional CPU activation offload.
- Packed heterogeneous samples with isolated attention and a microbatch token cap.
- Frozen-base LoRA streaming and full-training CPU parameter masters.
- Optional FP32 CPU gradient accumulation with fused Adafactor updates.
- Eager or tiled Triton weight-only INT8 computation, with optional ConvRot.

Choose memory settings for the target device, host memory, sequence lengths,
reference count, precision, and optimizer. CPU offload requires host memory and
transfer bandwidth. Checkpoint writing also consumes host memory and storage I/O.

## Training integrity

Resume validates optimizer settings, training state, cache identity, and LoRA
base/quantization settings. Warm starts initialize a fresh optimizer and schedule.
Checkpoints are staged before publication; tensor streaming bounds export workspace.
Cache fingerprints cover source inputs, encoder assets, and preprocessing settings.
Image preprocessing applies EXIF orientation. Held-out validation uses fixed noise
levels and separate randomness without changing the training RNG state.

## Verification scope

CPU and CUDA regression checks cover packed attention isolation, activation
layouts, gradient accumulation, quantized forward/backward computation, optimizer
updates, cache provenance, deterministic validation, and checkpoint save/resume.
Released-model execution checks cover packed LoRA and full-training updates and
checkpoint readability. Execution checks do not establish dataset convergence,
image quality, universal memory requirements, or steady-state throughput.
Learning rates require dataset-specific tuning. Measure checkpoint I/O separately.
