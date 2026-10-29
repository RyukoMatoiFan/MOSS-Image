"""Optional Triton kernels for frozen weight-only INT8 linear layers.

The kernels decode each INT8 weight tile inside the matrix multiplication.  No
full floating-point weight is constructed.  ConvRot uses the same GEMM kernels
and a staged Hadamard transform over bounded row chunks.
"""

from __future__ import annotations

from typing import Literal

import torch

QuantizedBackend = Literal["auto", "eager", "triton"]

_BACKENDS = {"auto", "eager", "triton"}
_CONVROT_WORKSPACE_ELEMENTS = 256 * 1024

try:
    import triton
    import triton.language as tl
except ImportError as error:  # Triton is an optional CUDA-only dependency.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Exception | None = error
else:
    _TRITON_IMPORT_ERROR = None


def validate_quantized_backend(backend: str) -> QuantizedBackend:
    if backend not in _BACKENDS:
        raise ValueError("quantized backend must be auto, eager or triton.")
    return backend  # type: ignore[return-value]


def triton_is_available() -> bool:
    """Return whether the optional Triton package and a CUDA runtime are present."""
    return triton is not None and torch.cuda.is_available()


def _unsupported_reason(tokens: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> str | None:
    if triton is None:
        detail = f" ({_TRITON_IMPORT_ERROR})" if _TRITON_IMPORT_ERROR is not None else ""
        return f"Triton is not installed{detail}"
    if tokens.device.type != "cuda":
        return "the input is not on CUDA"
    if weight.device != tokens.device or scale.device != tokens.device:
        return "the quantized weight and scale are not on the input CUDA device"
    if tokens.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return f"input dtype {tokens.dtype} is unsupported"
    if tokens.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return "the CUDA device does not support bfloat16"
    if weight.dtype != torch.int8 or scale.dtype != torch.float32:
        return "weights must be int8 and scales must be float32"
    if tokens.ndim < 1 or tokens.shape[-1] != weight.shape[1]:
        return "input and weight dimensions do not match"
    if not tokens.is_contiguous() or not weight.is_contiguous() or not scale.is_contiguous():
        return "input, weight and scale must be contiguous"
    if tokens.numel() == 0 or weight.numel() == 0:
        return "empty matrices are unsupported"
    return None


def resolve_quantized_backend(
    backend: str,
    tokens: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> QuantizedBackend:
    """Resolve ``auto`` and validate an explicitly requested Triton execution."""
    selected = validate_quantized_backend(backend)
    if selected == "eager":
        return selected
    reason = _unsupported_reason(tokens, weight, scale)
    if selected == "auto":
        return "eager" if reason is not None else "triton"
    if reason is not None:
        raise RuntimeError(f"Triton quantized backend is unavailable: {reason}.")
    return selected


def autocast_compute_dtype(tokens: torch.Tensor, module_dtype: torch.dtype) -> torch.dtype:
    """Match CUDA autocast's output dtype for a linear operation."""
    if tokens.device.type != "cuda":
        return module_dtype
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch before the device-specific overload.
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return module_dtype
    try:
        return torch.get_autocast_dtype("cuda")
    except AttributeError:
        return torch.get_autocast_gpu_dtype()


if triton is not None:

    @triton.jit
    def _int8_linear_forward_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        VALUE_KIND: tl.constexpr,
        WEIGHT_KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program = tl.program_id(0)
        blocks_n = tl.cdiv(out_features, BLOCK_N)
        block_m = program // blocks_n
        block_n = program % blocks_n
        offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for start_k in range(0, in_features, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            values = tl.load(
                x_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
                other=0.0,
            )
            quantized = tl.load(
                weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_n[:, None] < out_features) & (offsets_k[None, :] < in_features),
                other=0,
            )
            scales = tl.load(scale_ptr + offsets_n, mask=offsets_n < out_features, other=0.0)
            if WEIGHT_KIND == 0:
                decoded = (quantized.to(tl.float16) * scales.to(tl.float16)[:, None]).to(tl.float16)
            elif WEIGHT_KIND == 1:
                decoded = (quantized.to(tl.bfloat16) * scales.to(tl.bfloat16)[:, None]).to(tl.bfloat16)
            else:
                decoded = quantized.to(tl.float32) * scales[:, None]
            if VALUE_KIND == 0:
                values = values.to(tl.float16)
                decoded = decoded.to(tl.float16)
            elif VALUE_KIND == 1:
                values = values.to(tl.bfloat16)
                decoded = decoded.to(tl.bfloat16)
            accumulator += tl.dot(values, tl.trans(decoded), input_precision="ieee")

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < out_features, other=0.0)
            if VALUE_KIND == 0:
                bias = bias.to(tl.float16)
            elif VALUE_KIND == 1:
                bias = bias.to(tl.bfloat16)
            accumulator += bias.to(tl.float32)[None, :]
        if VALUE_KIND == 0:
            accumulator = accumulator.to(tl.float16)
        elif VALUE_KIND == 1:
            accumulator = accumulator.to(tl.bfloat16)
        tl.store(
            output_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
        )


    @triton.jit
    def _int8_linear_backward_kernel(
        gradient_ptr,
        weight_ptr,
        scale_ptr,
        grad_input_ptr,
        rows,
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        VALUE_KIND: tl.constexpr,
        WEIGHT_KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        program = tl.program_id(0)
        blocks_k = tl.cdiv(in_features, BLOCK_K)
        block_m = program // blocks_k
        block_k = program % blocks_k
        offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)

        for start_n in range(0, out_features, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            gradient = tl.load(
                gradient_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
                other=0.0,
            )
            quantized = tl.load(
                weight_ptr + offsets_n[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_n[:, None] < out_features) & (offsets_k[None, :] < in_features),
                other=0,
            )
            scales = tl.load(scale_ptr + offsets_n, mask=offsets_n < out_features, other=0.0)
            if WEIGHT_KIND == 0:
                decoded = (quantized.to(tl.float16) * scales.to(tl.float16)[:, None]).to(tl.float16)
            elif WEIGHT_KIND == 1:
                decoded = (quantized.to(tl.bfloat16) * scales.to(tl.bfloat16)[:, None]).to(tl.bfloat16)
            else:
                decoded = quantized.to(tl.float32) * scales[:, None]
            if VALUE_KIND == 0:
                gradient = gradient.to(tl.float16)
                decoded = decoded.to(tl.float16)
            elif VALUE_KIND == 1:
                gradient = gradient.to(tl.bfloat16)
                decoded = decoded.to(tl.bfloat16)
            accumulator += tl.dot(gradient, decoded, input_precision="ieee")

        if VALUE_KIND == 0:
            accumulator = accumulator.to(tl.float16)
        elif VALUE_KIND == 1:
            accumulator = accumulator.to(tl.bfloat16)
        tl.store(
            grad_input_ptr + offsets_m[:, None] * in_features + offsets_k[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
        )


    @triton.jit
    def _hadamard_h4_stage_kernel(
        input_ptr,
        output_ptr,
        elements,
        width: tl.constexpr,
        STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        columns = offsets % width
        rows = offsets // width
        digit = (columns // STRIDE) % 4
        base = columns - columns % (4 * STRIDE)
        position = columns % STRIDE
        source = rows * width + base + position
        a = tl.load(input_ptr + source, mask=valid, other=0.0).to(tl.float32)
        b = tl.load(input_ptr + source + STRIDE, mask=valid, other=0.0).to(tl.float32)
        c = tl.load(input_ptr + source + 2 * STRIDE, mask=valid, other=0.0).to(tl.float32)
        d = tl.load(input_ptr + source + 3 * STRIDE, mask=valid, other=0.0).to(tl.float32)
        total = a + b + c + d
        negative = tl.where(digit == 0, d, tl.where(digit == 1, c, tl.where(digit == 2, b, a)))
        tl.store(output_ptr + offsets, (total - 2.0 * negative) * 0.5, mask=valid)


    @triton.jit
    def _hadamard_h2_stage_kernel(
        input_ptr,
        output_ptr,
        elements,
        width: tl.constexpr,
        STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        columns = offsets % width
        rows = offsets // width
        digit = (columns // STRIDE) % 2
        base = columns - columns % (2 * STRIDE)
        position = columns % STRIDE
        source = rows * width + base + position
        a = tl.load(input_ptr + source, mask=valid, other=0.0).to(tl.float32)
        b = tl.load(input_ptr + source + STRIDE, mask=valid, other=0.0).to(tl.float32)
        value = tl.where(digit == 0, a + b, a - b) * 0.7071067811865476
        tl.store(output_ptr + offsets, value, mask=valid)


def _value_kind(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    raise TypeError(f"Triton quantized kernels do not support {dtype}.")


def _matrix_view(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.reshape(-1, tokens.shape[-1])


def triton_int8_linear_forward(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run tiled weight-only INT8 GEMM; all arguments must already be validated."""
    matrix = _matrix_view(tokens)
    rows, in_features = matrix.shape
    out_features = weight.shape[0]
    if out is None:
        output = torch.empty((rows, out_features), device=tokens.device, dtype=output_dtype)
    else:
        output = out.reshape(rows, out_features)
        if output.dtype != output_dtype or not output.is_contiguous():
            raise ValueError("Triton output buffer has the wrong dtype or is not contiguous.")
    if bias is not None and (bias.device != tokens.device or not bias.is_contiguous()):
        raise RuntimeError("Triton quantized bias must be contiguous and on the input device.")
    grid = (triton.cdiv(rows, 32) * triton.cdiv(out_features, 32),)
    _int8_linear_forward_kernel[grid](
        matrix,
        weight,
        scale,
        bias if bias is not None else output,
        output,
        rows,
        out_features,
        in_features,
        HAS_BIAS=bias is not None,
        VALUE_KIND=_value_kind(output_dtype),
        WEIGHT_KIND=_value_kind(weight_dtype),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*tokens.shape[:-1], out_features)


def triton_int8_linear_backward(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_shape: torch.Size,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute only the input gradient for a frozen quantized weight."""
    matrix = gradient.contiguous().reshape(-1, gradient.shape[-1])
    rows, out_features = matrix.shape
    in_features = weight.shape[1]
    if out is None:
        grad_input = torch.empty((rows, in_features), device=gradient.device, dtype=input_dtype)
    else:
        grad_input = out.reshape(rows, in_features)
        if grad_input.dtype != input_dtype or not grad_input.is_contiguous():
            raise ValueError("Triton input-gradient buffer has the wrong dtype or is not contiguous.")
    grid = (triton.cdiv(rows, 32) * triton.cdiv(in_features, 32),)
    _int8_linear_backward_kernel[grid](
        matrix,
        weight,
        scale,
        grad_input,
        rows,
        out_features,
        in_features,
        VALUE_KIND=_value_kind(gradient.dtype),
        WEIGHT_KIND=_value_kind(weight_dtype),
        BLOCK_M=32,
        BLOCK_K=32,
        BLOCK_N=32,
        num_warps=4,
    )
    return grad_input.reshape(input_shape)


def _hadamard_stages(group_size: int) -> list[tuple[int, int]]:
    stages = []
    stride = 1
    while stride * 4 <= group_size:
        stages.append((4, stride))
        stride *= 4
    if stride < group_size:
        stages.append((2, stride))
    return stages


def triton_hadamard_transform(
    tokens: torch.Tensor,
    group_size: int,
    *,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply grouped normalized ConvRot Hadamard stages on contiguous CUDA data."""
    if tokens.ndim != 2 or not tokens.is_contiguous():
        raise ValueError("Triton Hadamard input must be a contiguous matrix.")
    if tokens.shape[1] % group_size:
        raise ValueError("Input width must be divisible by Hadamard group size.")
    if out is None:
        out = torch.empty_like(tokens)
    if out.shape != tokens.shape or out.dtype != tokens.dtype or not out.is_contiguous():
        raise ValueError("Triton Hadamard output must match its contiguous input.")
    stages = _hadamard_stages(group_size)
    if not stages:
        out.copy_(tokens)
        return out
    if len(stages) > 1:
        if workspace is None:
            workspace = torch.empty_like(tokens)
        if workspace.shape != tokens.shape or workspace.dtype != tokens.dtype or not workspace.is_contiguous():
            raise ValueError("Triton Hadamard workspace must match its contiguous input.")
    source = tokens
    elements = tokens.numel()
    grid = (triton.cdiv(elements, 256),)
    for index, (radix, stride) in enumerate(stages):
        destination = out if (len(stages) - index) % 2 else workspace
        if radix == 4:
            _hadamard_h4_stage_kernel[grid](
                source, destination, elements, tokens.shape[1], STRIDE=stride, BLOCK=256, num_warps=4
            )
        else:
            _hadamard_h2_stage_kernel[grid](
                source, destination, elements, tokens.shape[1], STRIDE=stride, BLOCK=256, num_warps=4
            )
        source = destination
    return out


def _convrot_chunk_rows(in_features: int, rows: int) -> int:
    return min(rows, max(1, _CONVROT_WORKSPACE_ELEMENTS // in_features))


def triton_convrot_forward(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    """Run ConvRot through bounded Hadamard workspaces and tiled INT8 GEMM."""
    matrix = _matrix_view(tokens)
    rows, in_features = matrix.shape
    out_features = weight.shape[0]
    output = torch.empty((rows, out_features), device=tokens.device, dtype=output_dtype)
    chunk_rows = _convrot_chunk_rows(in_features, rows)
    rotated = torch.empty((chunk_rows, in_features), device=tokens.device, dtype=tokens.dtype)
    scratch = torch.empty_like(rotated) if len(_hadamard_stages(group_size)) > 1 else None
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        source = matrix[start : start + count]
        rotated_chunk = rotated[:count]
        scratch_chunk = None if scratch is None else scratch[:count]
        triton_hadamard_transform(source, group_size, out=rotated_chunk, workspace=scratch_chunk)
        gemm_input = rotated_chunk if rotated_chunk.dtype == weight_dtype else rotated_chunk.to(weight_dtype)
        triton_int8_linear_forward(
            gemm_input,
            weight,
            scale,
            bias,
            output_dtype,
            weight_dtype,
            out=output[start : start + count],
        )
    return output.reshape(*tokens.shape[:-1], out_features)


def triton_convrot_backward(
    gradient: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_shape: torch.Size,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    """Compute the frozen ConvRot input gradient with bounded workspaces."""
    matrix = gradient.contiguous().reshape(-1, gradient.shape[-1])
    rows = matrix.shape[0]
    in_features = weight.shape[1]
    grad_input = torch.empty((rows, in_features), device=gradient.device, dtype=input_dtype)
    chunk_rows = _convrot_chunk_rows(in_features, rows)
    rotated = torch.empty((chunk_rows, in_features), device=gradient.device, dtype=input_dtype)
    scratch = torch.empty_like(rotated) if len(_hadamard_stages(group_size)) > 1 else None
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        rotated_chunk = rotated[:count]
        triton_int8_linear_backward(
            matrix[start : start + count],
            weight,
            scale,
            torch.Size((count, in_features)),
            input_dtype,
            weight_dtype,
            out=rotated_chunk,
        )
        triton_hadamard_transform(
            rotated_chunk,
            group_size,
            out=grad_input[start : start + count],
            workspace=None if scratch is None else scratch[:count],
        )
    return grad_input.reshape(input_shape)


def convrot_workspace_bound_bytes(in_features: int, dtype: torch.dtype) -> int:
    """Upper bound for the two temporary ConvRot transform buffers."""
    if in_features < 1:
        raise ValueError("input width must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    rows = max(1, _CONVROT_WORKSPACE_ELEMENTS // in_features)
    return 2 * rows * in_features * element_size
