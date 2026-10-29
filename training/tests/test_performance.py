"""CPU validation for optional model fast paths, plus a manual GPU parity test."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

import training.performance as performance
from training.config import TrainConfig, load_config
from training.lora import LoRALinear
from training.model import Block, attention


def _lora() -> LoRALinear:
    torch.manual_seed(10)
    module = LoRALinear(nn.Linear(8, 6), rank=3, alpha=6.0)
    with torch.no_grad():
        module.lora_A.normal_(std=0.2)
        module.lora_B.normal_(std=0.2)
    return module


def test_lora_without_autocast_keeps_fp32_reference_math_and_gradients() -> None:
    module = _lora()
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    expected_tokens = tokens.detach().clone().requires_grad_(True)
    expected = module.base(expected_tokens) + F.linear(
        F.linear(expected_tokens, module.lora_A), module.lora_B
    ) * module.scale

    actual = module(tokens)
    actual.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tokens.grad, expected_tokens.grad)
    assert module.lora_A.dtype == torch.float32
    assert module.lora_B.dtype == torch.float32


def test_lora_autocast_casts_small_weights_instead_of_upcasting_tokens(monkeypatch) -> None:
    module = _lora()
    tokens = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    calls: list[tuple[torch.dtype, torch.dtype]] = []
    linear = F.linear

    def record(input_tensor, weight, bias=None):
        calls.append((input_tensor.dtype, weight.dtype))
        return linear(input_tensor, weight, bias)

    monkeypatch.setattr("training.lora.F.linear", record)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(tokens)
        output.float().square().mean().backward()

    # The first call is nn.Linear's frozen FP32 base under autocast. Only the two
    # adapter calls are owned by LoRALinear's fast path.
    assert calls[-2:] == [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ]
    assert module.lora_A.dtype == torch.float32
    assert module.lora_B.dtype == torch.float32
    assert module.lora_A.grad is not None and module.lora_A.grad.dtype == torch.float32
    assert module.lora_B.grad is not None and module.lora_B.grad.dtype == torch.float32


def test_auto_attention_is_existing_sdpa_on_cpu() -> None:
    torch.manual_seed(2)
    query = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    key = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    value = torch.randn(2, 3, 5, 4)
    expected = F.scaled_dot_product_attention(query.float(), key.float(), value)
    actual = performance.scaled_attention(query, key, value, "auto")
    torch.testing.assert_close(actual, expected)


def test_model_attention_forwards_layout_and_backend(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], str]] = []

    def fake_scaled(query, key, value, backend):
        assert query.shape == key.shape == value.shape
        calls.append((query.shape, backend))
        return value

    monkeypatch.setattr("training.model.scaled_attention", fake_scaled)
    query = torch.randn(1, 5, 32)
    key = torch.randn(1, 5, 32)
    value = torch.randn(1, 5, 32)
    rope = torch.cat((torch.zeros(1, 5, 4), torch.ones(1, 5, 4)), -1)
    output = attention(query, key, value, rope, heads=4, backend="sdpa")

    assert calls == [((1, 4, 5, 8), "sdpa")]
    assert output.shape == value.shape


def test_real_block_forward_dispatches_to_eager_core_without_schema_changes() -> None:
    torch.manual_seed(4)
    block = Block(dim=32, ffn_dim=64, heads=4).eval()
    before = tuple(block.state_dict())
    tokens = torch.randn(1, 5, 32)
    rope = torch.cat((torch.zeros(1, 5, 4), torch.ones(1, 5, 4)), -1)
    modulation = torch.randn(1, 6, 32)

    expected = block._forward_impl(tokens, rope, modulation)
    actual = block(tokens, rope, modulation)

    torch.testing.assert_close(actual, expected)
    assert tuple(block.state_dict()) == before


def test_flash_attention_import_failure_is_actionable(monkeypatch) -> None:
    performance._load_flash_attn_func.cache_clear()

    def missing(_name):
        raise ModuleNotFoundError("flash_attn")

    monkeypatch.setattr(performance.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="matching this Python, PyTorch, and CUDA"):
        performance.validate_attention_backend("flash_attn_2")
    performance._load_flash_attn_func.cache_clear()


def test_mocked_flash_attention_receives_bnhd_views(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, bool, bool]] = []

    def fake_flash(query, key, value, *, dropout_p, causal):
        calls.append((query.shape, key.shape, value.shape, query.data_ptr(), query.is_contiguous(), causal))
        assert dropout_p == 0.0
        return query + key + value

    monkeypatch.setattr(performance, "validate_attention_backend", lambda *args, **kwargs: "flash_attn_2")
    monkeypatch.setattr(performance, "_load_flash_attn_func", lambda: fake_flash)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 3, 5, 4)
    value = torch.randn(2, 3, 5, 4)
    output = performance.scaled_attention(query, key, value, "flash_attn_2")

    assert calls == [((2, 5, 3, 4), (2, 5, 3, 4), (2, 5, 3, 4), query.data_ptr(), False, False)]
    torch.testing.assert_close(output, query + key + value)


def test_explicit_gpu_backends_reject_cpu_without_cuda_queries(monkeypatch) -> None:
    fake_package = types.SimpleNamespace(flash_attn_func=lambda *args, **kwargs: None)
    performance._load_flash_attn_func.cache_clear()
    monkeypatch.setattr(performance.importlib, "import_module", lambda _name: fake_package)
    with pytest.raises(ValueError, match="requires CUDA"):
        performance.validate_attention_backend("flash_attn_2", device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        performance.validate_attention_backend("sdpa_cudnn", device="cpu", dtype=torch.bfloat16)
    performance._load_flash_attn_func.cache_clear()


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4))
        self.heads = 1

    def _forward_impl(self, tokens: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return tokens + self.weight * scale

    def forward(self, tokens: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return performance.model_performance_context(self).call_block(self, tokens, scale)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.adapter = _lora()


class _LegacyBlock(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 1


def test_shared_block_compile_is_one_plain_callable_and_preserves_schema(monkeypatch) -> None:
    model = _Model()
    before = tuple(model.state_dict())
    compile_calls: list[tuple[object, str]] = []

    def fake_compile(function, *, mode):
        compile_calls.append((function, mode))
        return lambda block, *args: function(block, *args)

    monkeypatch.setattr(performance, "validate_compile_environment", lambda: ("2.9.1", "3.5.1"))
    monkeypatch.setattr(torch, "compile", fake_compile)
    context = performance.configure_model_performance(
        model,
        attention_backend="auto",
        compile_blocks=True,
        compile_mode="default",
        lora_compute_dtype="bf16",
    )

    assert compile_calls == [(_Block._forward_impl, "default")]
    assert context.blocks_compiled
    assert model.adapter.lora_compute_dtype == torch.bfloat16
    assert all(layer.attention_backend == "auto" for layer in model.layers)
    assert tuple(model.state_dict()) == before
    assert "_moss_performance" not in dict(model.named_modules())
    assert performance.model_performance_context(model) is context
    assert all(performance.model_performance_context(layer) is context for layer in model.layers)
    tokens, scale = torch.randn(4), torch.randn(())
    torch.testing.assert_close(
        performance.call_model_block(model, model.layers[1], tokens, scale),
        model.layers[1](tokens, scale),
    )
    parameters = dict(model.layers[0].named_parameters())
    torch.testing.assert_close(
        functional_call(model.layers[0], parameters, (tokens, scale), strict=True),
        model.layers[0](tokens, scale),
    )


def test_eager_context_and_validation_are_functional() -> None:
    model = _Model()
    assert not performance.model_performance_context(model).blocks_compiled
    tokens, scale = torch.randn(4), torch.randn(())
    torch.testing.assert_close(
        performance.call_model_block(model, model.layers[0], tokens, scale),
        model.layers[0](tokens, scale),
    )
    with pytest.raises(ValueError, match="reduce-overhead"):
        performance.configure_model_performance(model, compile_blocks=True, compile_mode="reduce-overhead")

    legacy = _LegacyBlock()
    probe = torch.randn(4)
    torch.testing.assert_close(performance._DEFAULT_CONTEXT.call_block(legacy, probe), legacy(probe))


def test_compile_environment_reports_missing_and_mismatched_triton(monkeypatch) -> None:
    def missing(_distribution):
        raise performance.metadata.PackageNotFoundError

    monkeypatch.setattr(performance.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="no Triton distribution"):
        performance.validate_compile_environment()

    monkeypatch.setattr(performance.metadata, "version", lambda _distribution: "3.7.1.post27")
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu126")
    with pytest.raises(RuntimeError, match=r"requires .*>=3.5,<3.6"):
        performance.validate_compile_environment()


def test_performance_config_defaults_and_yaml_fields(tmp_path) -> None:
    defaults = TrainConfig()
    assert defaults.attention_backend == "auto"
    assert defaults.compile_blocks is False
    assert defaults.compile_mode == "default"
    assert defaults.lora_compute_dtype is None
    assert defaults.full_parameter_dtype == "fp32"

    path = tmp_path / "performance.yaml"
    path.write_text(
        "attention_backend: flash_attn_2\n"
        "compile_blocks: true\n"
        "compile_mode: max-autotune-no-cudagraphs\n"
        "lora_compute_dtype: bf16\n"
        "full_parameter_dtype: bf16\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.attention_backend == "flash_attn_2"
    assert loaded.compile_blocks is True
    assert loaded.compile_mode == "max-autotune-no-cudagraphs"
    assert loaded.lora_compute_dtype == "bf16"
    assert loaded.full_parameter_dtype == "bf16"


@pytest.mark.parametrize("preset", ["t2i_full_fast.yaml", "edit_full_fast.yaml"])
def test_shipped_full_fast_presets_accept_async_full(preset) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / preset
    config = load_config(path)
    assert config.mode == "full"
    assert config.optimizer == "adafactor_fused"
    assert config.block_swap_mode == "async_full"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_backend", "magic", "attention_backend"),
        ("block_swap_mode", "magic", "block_swap_mode"),
        ("compile_blocks", "maybe", "compile_blocks"),
        ("compile_mode", "reduce-overhead", "reduce-overhead"),
        ("lora_compute_dtype", "int8", "lora_compute_dtype"),
        ("full_parameter_dtype", "fp16", "full_parameter_dtype"),
    ],
)
def test_performance_config_rejects_invalid_values(tmp_path, field, value, message) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"{field}: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.gpu
def test_flash_attention_cuda_forward_backward_parity() -> None:
    """CUDA forward/backward parity for FlashAttention and SDPA."""

    if not torch.cuda.is_available():
        pytest.skip("requires an isolated CUDA GPU")
    try:
        performance.validate_attention_backend("flash_attn_2")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(3)
    query = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = performance.scaled_attention(query, key, value, "sdpa")
    actual = performance.scaled_attention(query, key, value, "flash_attn_2")
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
    reference.float().sum().backward(retain_graph=True)
    reference_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())
    query.grad = key.grad = value.grad = None
    actual.float().sum().backward()
    for gradient, expected in zip((query.grad, key.grad, value.grad), reference_grads):
        torch.testing.assert_close(gradient, expected, atol=3e-2, rtol=3e-2)
