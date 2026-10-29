"""CPU coverage for packed batches, selective checkpointing, and new presets."""

from __future__ import annotations

import copy

import pytest
import torch

import training.model as model_module
import training.performance as performance
from training.activation_offload import _offload_view, _restore_layout
from training.lora import inject_lora, lora_spec, save_lora, target_linear_names
from training.model import MOSSImageTransformer, _checkpointed_layer_indices
from training.performance import configure_model_performance, scaled_attention_varlen


@pytest.fixture(autouse=True)
def _forbid_cuda_initialization(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    if hasattr(torch.cuda, "_lazy_init"):
        monkeypatch.setattr(
            torch.cuda,
            "_lazy_init",
            lambda: (_ for _ in ()).throw(AssertionError("CPU tests must not initialize CUDA")),
        )


def _tiny(*, layers: int = 2, checkpointing: bool = False) -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=layers,
        text_dim=16,
        freq_dim=8,
        latent_channels=4,
        gradient_checkpointing=checkpointing,
    )


def _grad_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def test_packed_heterogeneous_forward_and_gradients_match_serial() -> None:
    torch.manual_seed(31)
    serial = _tiny().train()
    packed = copy.deepcopy(serial).train()
    serial_targets = [_grad_clone(torch.randn(4, 2, 2)), _grad_clone(torch.randn(4, 1, 3))]
    packed_targets = [_grad_clone(value) for value in serial_targets]
    serial_contexts = [_grad_clone(torch.randn(2, 16)), _grad_clone(torch.randn(3, 16))]
    packed_contexts = [_grad_clone(value) for value in serial_contexts]
    serial_refs = [
        [_grad_clone(torch.randn(4, 1, 2)), _grad_clone(torch.randn(4, 2, 1))],
        [_grad_clone(torch.randn(4, 1, 1))],
    ]
    packed_refs = [[_grad_clone(value) for value in values] for values in serial_refs]
    serial_time = _grad_clone(torch.tensor([125.0, 875.0]))
    packed_time = _grad_clone(serial_time)

    serial_outputs = []
    for index, (target, context, references) in enumerate(
        zip(serial_targets, serial_contexts, serial_refs),
    ):
        output = serial(
            target.unsqueeze(0),
            serial_time[index : index + 1],
            context.unsqueeze(0),
            ref_hidden_states=[reference.unsqueeze(0) for reference in references],
        )
        serial_outputs.append(output.squeeze(0))
    packed_outputs = packed.forward_packed(packed_targets, packed_time, packed_contexts, packed_refs)

    serial_loss = sum(output.square().mean() for output in serial_outputs)
    packed_loss = sum(output.square().mean() for output in packed_outputs)
    serial_loss.backward()
    packed_loss.backward()

    for actual, expected in zip(packed_outputs, serial_outputs):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(packed_targets, serial_targets):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)
    for actual, expected in zip(packed_contexts, serial_contexts):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)
        assert actual.grad.abs().sum() > 0
    for actual_group, expected_group in zip(packed_refs, serial_refs):
        for actual, expected in zip(actual_group, expected_group):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-6)
            assert actual.grad.abs().sum() > 0
    torch.testing.assert_close(packed_time.grad, serial_time.grad, rtol=3e-5, atol=3e-6)
    assert packed_time.grad.abs().sum() > 0
    for (actual_name, actual), (expected_name, expected) in zip(
        packed.named_parameters(), serial.named_parameters(),
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=5e-6)


def test_varlen_sdpa_isolates_examples() -> None:
    torch.manual_seed(32)
    query = torch.randn(1, 2, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 8)
    cumulative = torch.tensor([0, 3, 7])
    actual = scaled_attention_varlen(query, key, value, cumulative, "sdpa")
    expected = torch.cat([
        torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
        )
        for start, end in ((0, 3), (3, 7))
    ], dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_varlen_flash_uses_packed_thd_layout(monkeypatch) -> None:
    calls = []

    def fake_flash(query, key, value, cu_q, cu_k, max_q, max_k, *, dropout_p, causal):
        calls.append((query.shape, cu_q.tolist(), max_q, dropout_p, causal))
        return value

    monkeypatch.setattr(performance, "validate_attention_backend", lambda *args, **kwargs: "flash_attn_2")
    monkeypatch.setattr(performance, "_load_flash_attn_varlen_func", lambda: fake_flash)
    query = torch.randn(1, 2, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 8)
    actual = scaled_attention_varlen(query, key, value, torch.tensor([0, 3, 7]), "flash_attn_2")
    assert calls == [((7, 2, 8), [0, 3, 7], 4, 0.0, False)]
    torch.testing.assert_close(actual, value)


def test_varlen_cudnn_fails_instead_of_falling_back_to_math(monkeypatch) -> None:
    monkeypatch.setattr(performance, "validate_attention_backend", lambda *args, **kwargs: "sdpa_cudnn")
    query = torch.randn(1, 2, 7, 8)
    with pytest.raises(ValueError, match="does not support packed"):
        scaled_attention_varlen(query, query, query, torch.tensor([0, 3, 7]), "sdpa_cudnn")


def test_selective_checkpointing_uses_deterministic_subset(monkeypatch) -> None:
    model = _tiny(layers=4, checkpointing=True).train()
    model.checkpoint_fraction = 0.5
    calls: list[int] = []
    real_checkpoint = model_module.checkpoint

    def record(function, *args, **kwargs):
        calls.append(next(index for index, layer in enumerate(model.layers) if layer is function))
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(model_module, "checkpoint", record)
    model(torch.randn(1, 4, 2, 2), torch.tensor([500.0]), torch.randn(1, 2, 16))
    assert calls == sorted(_checkpointed_layer_indices(4, 0.5))
    assert _checkpointed_layer_indices(4, 0.0) == frozenset()
    assert _checkpointed_layer_indices(4, 1.0) == frozenset(range(4))
    with pytest.raises(ValueError, match="between 0 and 1"):
        model.checkpoint_fraction = 1.1
        model(torch.randn(1, 4, 2, 2), torch.tensor([500.0]), torch.randn(1, 2, 16))


def test_packed_streamed_block_keeps_checkpoint_and_varlen_inputs() -> None:
    model = _tiny(layers=2, checkpointing=True).train()
    model.checkpoint_fraction = 0.0
    calls = []

    class Manager:
        def block(
            self,
            index,
            *inputs,
            forward_prefetch,
            backward_prefetch,
            checkpoint_block,
        ):
            calls.append((index, len(inputs), checkpoint_block, inputs[-1].tolist()))
            return model.layers[index](*inputs)

    model.blocks_to_swap = 1
    model.block_swap_manager = Manager()
    targets = [torch.randn(4, 2, 2), torch.randn(4, 1, 2)]
    contexts = [torch.randn(2, 16), torch.randn(3, 16)]
    outputs = model.forward_packed(targets, torch.tensor([200.0, 700.0]), contexts)
    sum(output.square().mean() for output in outputs).backward()
    assert calls == [(1, 4, True, [0, 6, 11])]


def test_activation_offload_is_cpu_noop() -> None:
    torch.manual_seed(33)
    base = _tiny().train()
    offloaded = copy.deepcopy(base).train()
    offloaded.activation_offload = True
    target = torch.randn(1, 4, 2, 2)
    context = torch.randn(1, 2, 16)
    expected = base(target, torch.tensor([250.0]), context)
    actual = offloaded(target, torch.tensor([250.0]), context)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()


def test_activation_layout_round_trip_preserves_padding_and_expansion() -> None:
    backing = torch.arange(64, dtype=torch.float32)
    padded = backing.as_strided((1, 1, 5, 5), (64, 64, 8, 1))
    view = _offload_view(padded)
    assert view is not None
    source, payload_shape = view
    restored = _restore_layout(
        source.contiguous(),
        torch.device("cpu"),
        tuple(padded.shape),
        tuple(padded.stride()),
        payload_shape,
        non_blocking=False,
    )
    assert restored.stride() == padded.stride()
    torch.testing.assert_close(restored, padded)

    expanded = torch.arange(5, dtype=torch.float32).view(1, 5).expand(3, 5)
    view = _offload_view(expanded)
    assert view is not None
    source, payload_shape = view
    assert payload_shape == (1, 5)
    restored = _restore_layout(
        source.contiguous(),
        torch.device("cpu"),
        tuple(expanded.shape),
        tuple(expanded.stride()),
        payload_shape,
        non_blocking=False,
    )
    assert restored.stride() == expanded.stride() == (0, 1)
    torch.testing.assert_close(restored, expanded)

    overlapping = backing.as_strided((3, 3), (1, 1))
    assert _offload_view(overlapping) is None


def test_lora_target_presets_and_round_trip_metadata(tmp_path) -> None:
    model = _tiny()
    joint = target_linear_names(model, "joint")
    joint_and_text = target_linear_names(model, "joint_and_text")
    legacy = target_linear_names(model, "blocks")
    blocks_and_io = target_linear_names(model, "blocks_and_io")
    assert joint and all(name.startswith("layers.") for name in joint)
    assert joint_and_text == legacy
    assert any(name.startswith("text_embedder.") for name in joint_and_text)
    assert set(blocks_and_io) == set(legacy) | {"video_embedder", "final_linear_video"}

    inject_lora(model, rank=2, alpha=3.0, targets="joint")
    path = tmp_path / "joint.safetensors"
    save_lora(model, path)
    assert lora_spec(path) == (2, 3.0, "joint")
    with pytest.raises(ValueError, match="joint_and_text"):
        target_linear_names(_tiny(), "unknown")


def test_compile_modes_that_enable_cuda_graphs_are_rejected() -> None:
    with pytest.raises(ValueError, match="CUDA graphs"):
        configure_model_performance(_tiny(), compile_mode="max-autotune")

