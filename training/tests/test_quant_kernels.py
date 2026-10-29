"""Backend selection and numerical checks for optional quantized kernels."""

import os

import pytest
import torch
from torch import nn

from training.convrot import FrozenConvRotLinear
from training.quant_kernels import convrot_workspace_bound_bytes, triton_is_available
from training.quantization import FrozenInt8Linear, configure_quantized_backend

_RUN_GPU_TESTS = os.environ.get("MOSS_RUN_GPU_TESTS") == "1"


def test_backend_selection_is_explicit_and_cpu_auto_preserves_eager_results():
    torch.manual_seed(12)
    base = nn.Linear(11, 7).requires_grad_(False)
    eager = FrozenInt8Linear(base)
    automatic = FrozenInt8Linear(base, backend="auto")
    automatic.load_state_dict(eager.state_dict())
    assert eager.quantized_backend == "eager"
    assert automatic.quantized_backend == "auto"

    tokens = torch.randn(2, 3, 11, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    output = automatic(tokens)
    expected = eager(reference)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(tokens.grad, reference.grad)


def test_module_backend_configuration_and_validation():
    model = nn.Sequential(
        FrozenInt8Linear(nn.Linear(8, 5).requires_grad_(False)),
        FrozenConvRotLinear(nn.Linear(5, 3).requires_grad_(False), group_size=4),
    )
    assert configure_quantized_backend(model, "auto") == ["0", "1"]
    assert all(layer.quantized_backend == "auto" for layer in model)
    model[0].set_quantized_backend("eager")
    assert model[0].quantized_backend == "eager"
    with pytest.raises(ValueError, match="auto, eager or triton"):
        model[0].quantized_backend = "cuda"


def test_explicit_triton_fails_clearly_on_cpu():
    module = FrozenInt8Linear(nn.Linear(8, 5).requires_grad_(False), backend="triton")
    with pytest.raises(RuntimeError, match="Triton quantized backend is unavailable.*not on CUDA"):
        module(torch.randn(2, 8))


def test_convrot_workspace_is_bounded_independently_of_token_count():
    expected = 2 * 256 * 1024 * torch.empty((), dtype=torch.bfloat16).element_size()
    assert convrot_workspace_bound_bytes(64, torch.bfloat16) == expected
    assert convrot_workspace_bound_bytes(4096, torch.bfloat16) == expected


@pytest.mark.gpu
@pytest.mark.skipif(not _RUN_GPU_TESTS, reason="set MOSS_RUN_GPU_TESTS=1 on a reserved CUDA device")
@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_int8_uneven_output_and_input_gradient_parity(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    torch.manual_seed(73)
    device = torch.device("cuda")
    base = nn.Linear(70, 37, device=device, dtype=dtype).requires_grad_(False)
    eager = FrozenInt8Linear(base, backend="eager")
    fused = FrozenInt8Linear(base, backend="triton")
    fused.load_state_dict(eager.state_dict())
    tokens = torch.randn(2, 3, 70, device=device, dtype=dtype, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    actual = fused(tokens)
    expected = eager(reference)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(tokens.grad, reference.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _RUN_GPU_TESTS, reason="set MOSS_RUN_GPU_TESTS=1 on a reserved CUDA device")
@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_triton_bf16_autocast_preserves_eager_decode_rounding(base_dtype):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    device = torch.device("cuda")
    base = nn.Linear(70, 37, bias=True, device=device, dtype=base_dtype).requires_grad_(False)
    eager = FrozenInt8Linear(base, backend="eager")
    fused = FrozenInt8Linear(base, backend="triton")
    # This scale distinguishes BF16(scale) * BF16(q) from BF16(FP32(scale * q)).
    scale = 0.0002928110350837356
    with torch.no_grad():
        eager.weight_int8.zero_()
        eager.weight_int8[0, 0] = 83
        eager.weight_int8[1, 1] = -83
        eager.weight_scale.fill_(scale)
        eager.weight_int8[2:6, 2] = 1
        eager.weight_scale[2:6] = 1
        eager.bias.zero_()
        # FP32(bias) and BF16(bias) land on opposite sides of an output rounding boundary.
        eager.bias[0] = 0.08285745223138713
    fused.load_state_dict(eager.state_dict())
    tokens = torch.zeros(1, 70, device=device, dtype=base_dtype, requires_grad=True)
    with torch.no_grad():
        tokens[0, :2] = 1
    reference = tokens.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fused(tokens)
        expected = eager(reference)
    gradient = torch.zeros_like(actual)
    gradient[0, :2] = 1
    # These four BF16 terms sum to a non-BF16 value. FP32 inputs must still
    # receive eager's BF16 matmul-output rounding in the custom backward.
    gradient[0, 2:6] = torch.tensor(
        [-0.1962890625, -2.265625, -0.52734375, 1.546875],
        device=device,
        dtype=torch.bfloat16,
    )
    actual.backward(gradient)
    expected.backward(gradient)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(tokens.grad, reference.grad, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _RUN_GPU_TESTS, reason="set MOSS_RUN_GPU_TESTS=1 on a reserved CUDA device")
@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16, 64, 256])
def test_triton_convrot_bf16_uneven_output_and_input_gradient_parity(group_size):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    torch.manual_seed(91 + group_size)
    device = torch.device("cuda")
    base = nn.Linear(256, 37, device=device, dtype=torch.bfloat16).requires_grad_(False)
    eager = FrozenConvRotLinear(base, group_size=group_size, backend="eager")
    fused = FrozenConvRotLinear(base, group_size=group_size, backend="triton")
    fused.load_state_dict(eager.state_dict())
    tokens = torch.randn(2, 3, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    actual = fused(tokens)
    expected = eager(reference)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(tokens.grad, reference.grad, atol=3e-2, rtol=3e-2)
