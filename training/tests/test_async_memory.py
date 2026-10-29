"""Async frozen-base block swapping matches a fully resident LoRA model."""

from __future__ import annotations

import copy
import gc
from collections.abc import Iterable

import pytest
import torch
from torch import nn

from training.lora import LoRALinear, inject_lora
from training.memory import place_model
from training.model import MOSSImageTransformer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams and pinned memory"),
]


def _tiny(*, dtype: torch.dtype = torch.float32) -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=4,
        text_dim=16,
        freq_dim=8,
        latent_channels=4,
        gradient_checkpointing=True,
    ).to(dtype=dtype)


def _prepare_lora(quantization: str, dtype: torch.dtype) -> MOSSImageTransformer:
    model = _tiny(dtype=dtype)
    inject_lora(model, rank=2, alpha=4.0, targets="blocks_and_io")
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.03)
    if quantization == "int8":
        from training.quantization import quantize_frozen_linears

        assert quantize_frozen_linears(model)
    elif quantization == "int8_convrot":
        from training.convrot import quantize_frozen_convrot_linears

        assert quantize_frozen_convrot_linears(model)
    return model


def _layer_frozen_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = {
        f"parameters.{name}": parameter
        for name, parameter in model.layers.named_parameters()
        if not parameter.requires_grad
    }
    tensors.update({f"buffers.{name}": buffer for name, buffer in model.layers.named_buffers()})
    return tensors


def _trainables(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def _inputs(device: torch.device, dtype: torch.dtype) -> list[dict[str, object]]:
    return [
        {
            "hidden_states": torch.randn(1, 4, 4, 4, device=device, dtype=dtype),
            "timestep": torch.tensor([250.0], device=device),
            "encoder_hidden_states": torch.randn(1, 3, 16, device=device, dtype=dtype),
            "ref_hidden_states": [torch.randn(1, 4, 2, 4, device=device, dtype=dtype)],
        },
        {
            "hidden_states": torch.randn(1, 4, 3, 5, device=device, dtype=dtype),
            "timestep": torch.tensor([750.0], device=device),
            "encoder_hidden_states": torch.randn(1, 4, 16, device=device, dtype=dtype),
            "ref_hidden_states": [
                torch.randn(1, 4, 3, 2, device=device, dtype=dtype),
                torch.randn(1, 4, 2, 3, device=device, dtype=dtype),
            ],
        },
    ]


def _optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict[str, object]],
    *,
    autocast: bool,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    outputs = []
    for batch in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            output = model(**batch)
            loss = output.float().square().mean() / 2
        outputs.append(output.detach().cpu())
        loss.backward()
    parameters = _trainables(model)
    assert parameters
    assert all(parameter.device.type == "cuda" for parameter in parameters.values())
    assert all(
        parameter.grad is not None
        and parameter.grad.device.type == "cuda"
        and torch.isfinite(parameter.grad).all()
        for parameter in parameters.values()
    )
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in parameters.items()
        if name.endswith("lora_A")
    )
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in parameters.items()}
    optimizer.step()
    updated = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    return outputs, gradients, updated


def _assert_named_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("quantization", ["dense", "int8", "int8_convrot"])
def test_async_frozen_lora_matches_resident_training_and_eval(precision: str, quantization: str) -> None:
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    torch.manual_seed(83)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    resident = _prepare_lora(quantization, dtype).train()
    asynchronous = copy.deepcopy(resident).train()
    assert all(
        torch.count_nonzero(module.lora_B).item() > 0
        for module in asynchronous.modules()
        if isinstance(module, LoRALinear)
    )
    frozen_before = {name: tensor.detach().clone() for name, tensor in _layer_frozen_tensors(asynchronous).items()}
    adapter_ids = {name: id(parameter) for name, parameter in _trainables(asynchronous).items()}
    initial_adapters = {
        name: parameter.detach().clone() for name, parameter in _trainables(resident).items()
    }

    device = torch.device("cuda")
    place_model(resident, device, 0)
    place_model(asynchronous, device, len(asynchronous.layers), block_swap_mode="async")

    manager = asynchronous.block_swap_manager
    assert manager is not None
    async_adapters = _trainables(asynchronous)
    assert {name: id(parameter) for name, parameter in async_adapters.items()} == adapter_ids
    assert all(parameter.device.type == "cuda" for parameter in async_adapters.values())
    frozen_placed = _layer_frozen_tensors(asynchronous)
    assert frozen_placed.keys() == frozen_before.keys()
    assert all(
        tensor.device.type == "cpu" and (tensor.numel() == 0 or tensor.is_pinned())
        for tensor in frozen_placed.values()
    )
    int8_names = [name for name in frozen_placed if name.endswith("weight_int8")]
    assert bool(int8_names) == (quantization != "dense")
    assert all(frozen_placed[name].dtype == torch.int8 for name in int8_names)

    resident_optimizer = torch.optim.AdamW(_trainables(resident).values(), lr=2e-3)
    async_optimizer = torch.optim.AdamW(async_adapters.values(), lr=2e-3)
    assert all(
        parameter.device.type == "cuda"
        for group in async_optimizer.param_groups
        for parameter in group["params"]
    )
    assert {
        id(parameter)
        for group in async_optimizer.param_groups
        for parameter in group["params"]
    } == set(adapter_ids.values())
    atol, rtol = ((4e-3, 3e-2) if precision == "bf16" else (2e-5, 2e-3))

    for step in range(2):
        torch.manual_seed(200 + step)
        batches = _inputs(device, dtype)
        resident_result = _optimization_step(
            resident, resident_optimizer, batches, autocast=precision == "bf16"
        )
        async_result = _optimization_step(
            asynchronous, async_optimizer, batches, autocast=precision == "bf16"
        )
        for resident_output, async_output in zip(resident_result[0], async_result[0]):
            torch.testing.assert_close(async_output, resident_output, atol=atol, rtol=rtol)
        _assert_named_close(async_result[1], resident_result[1], atol=atol, rtol=rtol)
        _assert_named_close(async_result[2], resident_result[2], atol=atol, rtol=rtol)

        if step == 0:
            resident.eval()
            asynchronous.eval()
            negative_batch = dict(batches[1])
            negative_batch["encoder_hidden_states"] = -batches[1]["encoder_hidden_states"]
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
            ):
                resident_conditional = resident(**batches[1])
                resident_unconditional = resident(**negative_batch, skip_layers=(1, 3))
                async_conditional = asynchronous(**batches[1])
                async_unconditional = asynchronous(**negative_batch, skip_layers=(1, 3))
                resident_cfg = resident_unconditional + 3 * (
                    resident_conditional - resident_unconditional
                )
                async_cfg = async_unconditional + 3 * (async_conditional - async_unconditional)
            assert resident_cfg.shape == async_cfg.shape == batches[1]["hidden_states"].shape
            torch.testing.assert_close(async_conditional, resident_conditional, atol=atol, rtol=rtol)
            torch.testing.assert_close(async_unconditional, resident_unconditional, atol=atol, rtol=rtol)
            torch.testing.assert_close(async_cfg, resident_cfg, atol=atol * 3, rtol=rtol)
            resident.train()
            asynchronous.train()
            assert resident.training and asynchronous.training

    final_adapters = {name: parameter.detach().cpu() for name, parameter in _trainables(resident).items()}
    assert any(not torch.equal(final_adapters[name], initial_adapters[name]) for name in final_adapters)
    assert {name: id(parameter) for name, parameter in _trainables(asynchronous).items()} == adapter_ids
    assert 1 <= manager.peak_resident_blocks <= 2
    frozen_after = _layer_frozen_tensors(asynchronous)
    assert frozen_after.keys() == frozen_before.keys()
    for name, original in frozen_before.items():
        assert frozen_after[name].device.type == "cpu"
        assert frozen_after[name].numel() == 0 or frozen_after[name].is_pinned()
        assert torch.equal(frozen_after[name], original), name
    assert all(frozen_after[name].dtype == torch.int8 for name in int8_names)


def test_partial_async_swap_supports_two_live_graphs_and_one_backward() -> None:
    torch.manual_seed(97)
    resident = _prepare_lora("dense", torch.float32).train()
    asynchronous = copy.deepcopy(resident).train()
    device = torch.device("cuda")
    place_model(resident, device, 0)
    place_model(asynchronous, device, 2, block_swap_mode="async")

    for layer in asynchronous.layers[:2]:
        assert all(parameter.device.type == "cuda" for parameter in layer.parameters())
    for layer in asynchronous.layers[2:]:
        assert all(
            parameter.device.type == ("cuda" if parameter.requires_grad else "cpu")
            for parameter in layer.parameters()
        )
    resident_parameters = _trainables(resident)
    async_parameters = _trainables(asynchronous)
    resident_optimizer = torch.optim.AdamW(resident_parameters.values(), lr=2e-3)
    async_optimizer = torch.optim.AdamW(async_parameters.values(), lr=2e-3)
    batches = _inputs(device, torch.float32)

    results = []
    for model, optimizer in (
        (resident, resident_optimizer),
        (asynchronous, async_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        outputs = [model(**batch) for batch in batches]
        assert all(output.grad_fn is not None for output in outputs)
        sum(output.float().square().mean() / len(outputs) for output in outputs).backward()
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in _trainables(model).items()
        }
        optimizer.step()
        updated = {
            name: parameter.detach().cpu().clone()
            for name, parameter in _trainables(model).items()
        }
        results.append(([output.detach().cpu() for output in outputs], gradients, updated))

    for resident_output, async_output in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(async_output, resident_output, atol=2e-5, rtol=2e-3)
    _assert_named_close(results[1][1], results[0][1], atol=2e-5, rtol=2e-3)
    _assert_named_close(results[1][2], results[0][2], atol=2e-5, rtol=2e-3)
    assert 1 <= asynchronous.block_swap_manager.peak_resident_blocks <= 2


def test_grad_enabled_eval_forward_backpropagates_through_four_streamed_blocks() -> None:
    torch.manual_seed(101)
    resident = _prepare_lora("dense", torch.float32).eval()
    asynchronous = copy.deepcopy(resident).eval()
    device = torch.device("cuda")
    place_model(resident, device, 0)
    place_model(asynchronous, device, 4, block_swap_mode="async")
    batch = _inputs(device, torch.float32)[0]

    resident_output = resident(**batch)
    async_output = asynchronous(**batch)
    assert torch.is_grad_enabled() and resident_output.grad_fn is not None and async_output.grad_fn is not None
    resident_output.float().square().mean().backward()
    async_output.float().square().mean().backward()

    torch.testing.assert_close(async_output, resident_output, atol=2e-5, rtol=2e-3)
    resident_gradients = {
        name: parameter.grad.detach().cpu() for name, parameter in _trainables(resident).items()
    }
    async_gradients = {
        name: parameter.grad.detach().cpu() for name, parameter in _trainables(asynchronous).items()
    }
    _assert_named_close(async_gradients, resident_gradients, atol=2e-5, rtol=2e-3)


def test_autograd_grad_releases_streamed_blocks_for_next_iteration_and_teardown() -> None:
    torch.manual_seed(103)
    model = _prepare_lora("dense", torch.float32).train()
    device = torch.device("cuda")
    place_model(model, device, 4, block_swap_mode="async")
    manager = model.block_swap_manager
    parameters = _trainables(model)
    optimizer = torch.optim.AdamW(parameters.values(), lr=2e-3)
    batches = _inputs(device, torch.float32)

    first_loss = model(**batches[0]).float().square().mean()
    explicit_gradients = torch.autograd.grad(first_loss, tuple(parameters.values()))
    assert len(explicit_gradients) == len(parameters)
    assert all(gradient.device.type == "cuda" and torch.isfinite(gradient).all() for gradient in explicit_gradients)
    assert all(parameter.grad is None for parameter in parameters.values())

    optimizer.zero_grad(set_to_none=True)
    second_loss = model(**batches[1]).float().square().mean()
    second_loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.values())
    optimizer.step()

    place_model(model, torch.device("cpu"), 0)
    assert manager.closed
    assert manager.resident_block_indices == ()
    assert model.block_swap_manager is None
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_retained_graph_supports_two_backward_traversals() -> None:
    torch.manual_seed(107)
    model = _prepare_lora("dense", torch.float32).train()
    device = torch.device("cuda")
    place_model(model, device, 4, block_swap_mode="async")
    parameters = _trainables(model)
    loss = model(**_inputs(device, torch.float32)[0]).float().square().mean()

    loss.backward(retain_graph=True)
    first_gradients = {
        name: parameter.grad.detach().clone() for name, parameter in parameters.items()
    }
    loss.backward()
    for name, parameter in parameters.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, first_gradients[name] * 2, atol=2e-5, rtol=2e-3)


def test_recompute_exception_allows_cleanup_and_fresh_async_placement() -> None:
    torch.manual_seed(109)
    model = _prepare_lora("dense", torch.float32).train()
    device = torch.device("cuda")
    place_model(model, device, 4, block_swap_mode="async")
    failed_manager = model.block_swap_manager
    layer = model.layers[-1]
    original_forward = layer.forward
    calls = 0

    def fail_during_recompute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("recompute failure")
        return original_forward(*args, **kwargs)

    layer.forward = fail_during_recompute
    loss = model(**_inputs(device, torch.float32)[0]).float().square().mean()
    try:
        with pytest.raises(RuntimeError, match="recompute failure"):
            loss.backward()
    finally:
        layer.forward = original_forward
    assert calls == 2
    # Parameter storage cannot move while the deliberately failed graph still
    # owns tensor references. Real recovery likewise starts after dropping it.
    del loss
    gc.collect()

    place_model(model, torch.device("cpu"), 0)
    assert failed_manager.closed
    assert failed_manager.resident_block_indices == ()
    assert model.block_swap_manager is None
    place_model(model, device, 4, block_swap_mode="async")
    recovered_manager = model.block_swap_manager
    assert recovered_manager is not failed_manager
    model.zero_grad(set_to_none=True)
    recovered_loss = model(**_inputs(device, torch.float32)[1]).float().square().mean()
    recovered_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in _trainables(model).values()
    )
    place_model(model, torch.device("cpu"), 0)
    assert recovered_manager.closed


def test_cpu_placement_closes_manager_after_eval_prefetch() -> None:
    torch.manual_seed(113)
    model = _prepare_lora("dense", torch.float32).eval()
    device = torch.device("cuda")
    place_model(model, device, 4, block_swap_mode="async")
    manager = model.block_swap_manager
    assert manager.h2d_copy_count == 0
    with torch.no_grad():
        output = model(**_inputs(device, torch.float32)[0])
    assert output.is_cuda
    assert manager.resident_block_indices
    assert manager.h2d_copy_count == len(model.layers)

    place_model(model, torch.device("cpu"), 0)
    assert manager.closed
    assert manager.resident_block_indices == ()
    assert model.block_swap_manager is None
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())


def test_async_rejects_full_training_before_mutating_the_model() -> None:
    model = _tiny()
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="H2D-only.*frozen base weights"):
        place_model(model, torch.device("cuda"), 2, block_swap_mode="async")
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert not hasattr(model, "block_swap_manager")


def test_async_rejects_cpu_and_unknown_mode_before_mutating_the_model() -> None:
    model = _prepare_lora("dense", torch.float32)
    adapter_ids = {name: id(parameter) for name, parameter in _trainables(model).items()}
    with pytest.raises(ValueError, match="requires a CUDA device"):
        place_model(model, torch.device("cpu"), 2, block_swap_mode="async")
    with pytest.raises(ValueError, match="block_swap_mode must be 'sync' or 'async'"):
        place_model(model, torch.device("cpu"), 2, block_swap_mode="overlap")
    assert {name: id(parameter) for name, parameter in _trainables(model).items()} == adapter_ids
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert not hasattr(model, "block_swap_manager")
