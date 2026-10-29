"""CUDA parity gates for packed attention and activation offload.

These tests require CUDA and are separate from the CPU suite.
"""

from __future__ import annotations

import copy

import pytest
import torch

import training.model as model_module
from training.fused_adafactor import FusedAdafactor
from training.lora import LoRALinear, inject_lora
from training.memory import place_model
from training.model import MOSSImageTransformer, _checkpointed_layer_indices
from training.performance import configure_model_performance

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a reserved CUDA device"),
]


def _tiny(*, layers: int = 2, dtype: torch.dtype = torch.float32) -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=layers,
        text_dim=16,
        freq_dim=8,
        latent_channels=4,
        gradient_checkpointing=True,
    ).to(dtype=dtype)


def _leaf(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype).detach().requires_grad_(True)


def _inputs(device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(902)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    targets = [
        _leaf(random(4, 2, 2), device, dtype),
        _leaf(random(4, 1, 3), device, dtype),
    ]
    contexts = [
        _leaf(random(2, 16), device, dtype),
        _leaf(random(3, 16), device, dtype),
    ]
    references = [
        [_leaf(random(4, 1, 2), device, dtype), _leaf(random(4, 2, 1), device, dtype)],
        [_leaf(random(4, 1, 1), device, dtype)],
    ]
    timestep = _leaf(torch.tensor([125.0, 875.0]), device, torch.float32)
    return targets, timestep, contexts, references


def _clone_inputs(inputs):
    targets, timestep, contexts, references = inputs
    return (
        [value.detach().clone().requires_grad_(True) for value in targets],
        timestep.detach().clone().requires_grad_(True),
        [value.detach().clone().requires_grad_(True) for value in contexts],
        [[value.detach().clone().requires_grad_(True) for value in group] for group in references],
    )


def _packed_backward(model: MOSSImageTransformer, inputs):
    targets, timestep, contexts, references = inputs
    outputs = model.forward_packed(targets, timestep, contexts, references)
    sum(output.float().square().mean() for output in outputs).backward()
    parameter_grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    input_grads = (
        [value.grad.detach().cpu().clone() for value in targets],
        timestep.grad.detach().cpu().clone(),
        [value.grad.detach().cpu().clone() for value in contexts],
        [[value.grad.detach().cpu().clone() for value in group] for group in references],
    )
    return [output.detach().cpu() for output in outputs], parameter_grads, input_grads


def _serial_backward(model: MOSSImageTransformer, inputs):
    targets, timestep, contexts, references = inputs
    outputs = []
    for index, (target, context, sample_refs) in enumerate(zip(targets, contexts, references)):
        output = model(
            target.unsqueeze(0),
            timestep[index : index + 1],
            context.unsqueeze(0),
            ref_hidden_states=[reference.unsqueeze(0) for reference in sample_refs],
        )
        outputs.append(output.squeeze(0))
    sum(output.float().square().mean() for output in outputs).backward()
    parameter_grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    input_grads = (
        [value.grad.detach().cpu().clone() for value in targets],
        timestep.grad.detach().cpu().clone(),
        [value.grad.detach().cpu().clone() for value in contexts],
        [[value.grad.detach().cpu().clone() for value in group] for group in references],
    )
    return [output.detach().cpu() for output in outputs], parameter_grads, input_grads


def _assert_nested_close(actual, expected, *, rtol: float, atol: float) -> None:
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol, msg=key)
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_close(left, right, rtol=rtol, atol=atol)
    else:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", ["sdpa", "flash_attn_2"])
def test_packed_varlen_model_output_and_gradients_match_serial(backend: str) -> None:
    if backend == "flash_attn_2" and not torch.cuda.is_bf16_supported():
        pytest.skip("FlashAttention parity requires CUDA bf16 support")
    dtype = torch.bfloat16 if backend == "flash_attn_2" else torch.float32
    torch.manual_seed(901)
    serial = _tiny(dtype=dtype).train()
    packed = copy.deepcopy(serial).train()
    try:
        configure_model_performance(serial, attention_backend=backend)
        configure_model_performance(packed, attention_backend=backend)
    except RuntimeError as error:
        if backend == "flash_attn_2":
            pytest.skip(str(error))
        raise
    device = torch.device("cuda")
    serial.to(device)
    packed.to(device)
    inputs = _inputs(device, dtype)
    serial_result = _serial_backward(serial, inputs)
    packed_result = _packed_backward(packed, _clone_inputs(inputs))
    tolerance = (4e-2, 4e-2) if backend == "flash_attn_2" else (5e-5, 5e-6)
    _assert_nested_close(packed_result, serial_result, rtol=tolerance[0], atol=tolerance[1])


def test_checkpoint_fraction_half_recomputes_selected_cuda_blocks(monkeypatch) -> None:
    model = _tiny(layers=4).cuda().train()
    model.checkpoint_fraction = 0.5
    calls = []
    original = model_module.checkpoint

    def record(function, *args, **kwargs):
        calls.append(next(index for index, layer in enumerate(model.layers) if layer is function))
        return original(function, *args, **kwargs)

    monkeypatch.setattr(model_module, "checkpoint", record)
    outputs = model.forward_packed(*_inputs(torch.device("cuda"), torch.float32))
    sum(output.square().mean() for output in outputs).backward()
    assert calls == sorted(_checkpointed_layer_indices(4, 0.5))


@pytest.mark.parametrize("swap_mode", ["sync", "async"])
def test_activation_offload_matches_with_frozen_block_swapping(swap_mode: str) -> None:
    torch.manual_seed(903)
    baseline = _tiny(layers=3).train()
    if swap_mode == "async":
        inject_lora(baseline, rank=2, alpha=4.0, targets="blocks_and_io")
        with torch.no_grad():
            for module in baseline.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(std=0.03)
    offloaded = copy.deepcopy(baseline).train()
    offloaded.activation_offload = True
    device = torch.device("cuda")
    place_model(baseline, device, 2, block_swap_mode=swap_mode)
    place_model(offloaded, device, 2, block_swap_mode=swap_mode)
    baseline_manager = getattr(baseline, "block_swap_manager", None)
    offloaded_manager = getattr(offloaded, "block_swap_manager", None)
    try:
        inputs = _inputs(device, torch.float32)
        expected = _packed_backward(baseline, inputs)
        actual = _packed_backward(offloaded, _clone_inputs(inputs))
        _assert_nested_close(actual, expected, rtol=5e-5, atol=5e-6)
        if swap_mode == "async":
            assert offloaded_manager.owned_cuda_buffers()
    finally:
        if baseline_manager is not None:
            baseline_manager.close()
        if offloaded_manager is not None:
            offloaded_manager.close()


def test_packed_async_full_runs_one_trainable_streamed_graph() -> None:
    torch.manual_seed(904)
    cpu_model = _tiny(layers=3).train()
    resident = copy.deepcopy(cpu_model).cuda().eval()
    streamed = cpu_model.train()
    streamed.activation_offload = True
    place_model(streamed, torch.device("cuda"), 2, block_swap_mode="async_full")
    manager = streamed.block_swap_manager
    optimizer = FusedAdafactor(streamed.named_parameters(), lr=2e-3).install_hooks(manager)
    inputs = _inputs(torch.device("cuda"), torch.float32)
    reference_inputs = _clone_inputs(inputs)
    try:
        with torch.no_grad():
            expected = resident.forward_packed(*reference_inputs)
        optimizer.begin_step()
        optimizer.set_step_enabled(True)
        outputs = streamed.forward_packed(*inputs)
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=5e-5, atol=5e-6)
        sum(output.float().square().mean() for output in outputs).backward()
        optimizer.finish_backward()
        manager.finish_backward()
        assert optimizer.updated_parameter_count == len(list(streamed.parameters()))
        assert manager.d2h_copy_count == 2
        assert inputs[0][0].grad is not None
    finally:
        optimizer.close()
        manager.close()

