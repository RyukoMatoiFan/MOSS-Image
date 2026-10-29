"""Packed full-training ring validation with explicitly gated CUDA cases."""

import copy
import io

import pytest
import torch
from torch import nn

from training.fused_adafactor import FusedAdafactor
from training.memory import place_model
from training.model import MOSSImageTransformer
from training.trainable_offload import TrainableBlockStreamer


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)

    def forward(self, value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.up(self.norm(value)))
        return value + self.down(hidden) * scale


class _Tiny(nn.Module):
    def __init__(self, depth: int = 4, width: int = 8) -> None:
        super().__init__()
        self.time = nn.Linear(1, width)
        self.layers = nn.ModuleList([_Block(width) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)
        self.block_swap_manager = None
        self.blocks_to_swap = 0

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale = self.time(time).tanh().unsqueeze(1)
        streamed = list(range(len(self.layers)))
        for position, (index, layer) in enumerate(zip(streamed, self.layers)):
            if self.block_swap_manager is None:
                value = torch.utils.checkpoint.checkpoint(
                    layer, value, scale, use_reentrant=False
                )
            else:
                value = self.block_swap_manager.block(
                    index,
                    value,
                    scale,
                    forward_prefetch=streamed[position + 1] if position + 1 < len(streamed) else None,
                    backward_prefetch=streamed[position - 1] if position else None,
                    checkpoint_block=True,
                )
        return self.head(self.final_norm(value))


def _tiny_moss(*, layers: int = 3) -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=layers,
        text_dim=16,
        freq_dim=8,
        latent_channels=4,
        gradient_checkpointing=True,
    ).float()


def _moss_examples(seed: int) -> list[dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return [
        {
            "latent": random(1, 4, 2, 2),
            "context": random(1, 3, 16),
            "mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
            "references": [random(1, 4, 1, 2)],
            "target": random(1, 4, 2, 2),
            "timestep": 137.0,
        },
        {
            "latent": random(1, 4, 2, 3),
            "context": random(1, 5, 16),
            "mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            "references": [random(1, 4, 2, 1), random(1, 4, 1, 3)],
            "target": random(1, 4, 2, 3),
            "timestep": 811.0,
        },
    ]


def _cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _run_moss_logical_step(
    model: MOSSImageTransformer,
    optimizer: FusedAdafactor,
    examples: list[dict[str, object]],
) -> list[torch.Tensor]:
    manager = model.block_swap_manager
    device = next(model.video_embedder.parameters()).device
    total_count = sum(example["target"].numel() for example in examples)
    outputs: list[torch.Tensor] = []
    optimizer.begin_step()
    for index, example in enumerate(examples):
        last = index == len(examples) - 1
        optimizer.set_step_enabled(last)
        latent = example["latent"].to(device)
        context = example["context"].to(device)
        mask = example["mask"].to(device)
        references = [reference.to(device) for reference in example["references"]]
        target = example["target"].to(device)
        output = model(
            latent,
            example["timestep"],
            context,
            encoder_attention_mask=mask,
            ref_hidden_states=references,
        )
        outputs.append(output.detach().cpu())
        ((output - target).float().square().sum() / total_count).backward()
        if last:
            optimizer.finish_backward()
        if manager is not None:
            manager.finish_backward()
    return outputs


def _assert_model_states_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6, msg=name)


def _assert_optimizer_moments_close(
    actual: FusedAdafactor,
    expected: FusedAdafactor,
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state["fused_adafactor"] == expected_state["fused_adafactor"]
    assert actual_state["param_groups"] == expected_state["param_groups"]
    assert actual_state["state"].keys() == expected_state["state"].keys()
    for parameter_id, expected_values in expected_state["state"].items():
        actual_values = actual_state["state"][parameter_id]
        assert actual_values.keys() == expected_values.keys()
        for key, expected_value in expected_values.items():
            actual_value = actual_values[key]
            if isinstance(expected_value, torch.Tensor):
                assert actual_value.device.type == expected_value.device.type == "cuda"
                assert actual_value.dtype == expected_value.dtype == torch.float32
                torch.testing.assert_close(actual_value, expected_value, rtol=1e-6, atol=1e-7)
            else:
                assert actual_value == expected_value


@pytest.mark.gpu
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a reserved CUDA device")
def test_real_moss_ring_matches_resident_across_accumulated_logical_steps() -> None:
    torch.random.default_generator.manual_seed(101)
    resident = _tiny_moss(layers=3)
    streamed = copy.deepcopy(resident)
    resident_ids = {name: id(parameter) for name, parameter in resident.named_parameters()}
    streamed_ids = {name: id(parameter) for name, parameter in streamed.named_parameters()}
    initial_modulation = streamed.shared_block_modulation.weight.detach().clone()

    place_model(resident, torch.device("cuda"), 0, block_swap_mode="async_full")
    place_model(streamed, torch.device("cuda"), 2, block_swap_mode="async_full")
    manager = streamed.block_swap_manager
    assert isinstance(manager, TrainableBlockStreamer)
    resident_optimizer = FusedAdafactor(resident.named_parameters(), lr=2e-3).install_hooks()
    streamed_optimizer = FusedAdafactor(streamed.named_parameters(), lr=2e-3).install_hooks(manager)

    try:
        for step, seed in enumerate((211, 307), start=1):
            examples = _moss_examples(seed)
            resident_outputs = _run_moss_logical_step(resident, resident_optimizer, examples)
            streamed_outputs = _run_moss_logical_step(streamed, streamed_optimizer, examples)
            for actual, expected in zip(streamed_outputs, resident_outputs):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

            manager.commit()
            _assert_model_states_close(_cpu_model_state(streamed), _cpu_model_state(resident))
            assert resident_optimizer.global_step == streamed_optimizer.global_step == step
            parameter_count = len(list(streamed.parameters()))
            assert resident_optimizer.updated_parameter_count == parameter_count
            assert streamed_optimizer.updated_parameter_count == parameter_count
            assert all(parameter.grad is None for parameter in resident.parameters())
            assert all(parameter.grad is None for parameter in streamed.parameters())

        assert manager.d2h_copy_count == 4
        assert 1 <= manager.peak_resident_blocks <= 2
        assert manager.resident_block_indices == ()
        assert {
            name: id(parameter) for name, parameter in resident.named_parameters()
        } == resident_ids
        assert {
            name: id(parameter) for name, parameter in streamed.named_parameters()
        } == streamed_ids
        assert not torch.equal(
            streamed.shared_block_modulation.weight.detach().cpu(), initial_modulation
        )
        assert all(
            parameter.device.type == "cpu"
            for layer in streamed.layers[1:]
            for parameter in layer.parameters()
        )
    finally:
        resident_optimizer.close()
        streamed_optimizer.close()
        manager.close()


@pytest.mark.gpu
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a reserved CUDA device")
def test_real_moss_ring_optimizer_save_resume_matches_uninterrupted_step() -> None:
    torch.random.default_generator.manual_seed(401)
    uninterrupted = _tiny_moss(layers=2)
    place_model(uninterrupted, torch.device("cuda"), 2, block_swap_mode="async_full")
    uninterrupted_manager = uninterrupted.block_swap_manager
    assert isinstance(uninterrupted_manager, TrainableBlockStreamer)
    uninterrupted_optimizer = FusedAdafactor(
        uninterrupted.named_parameters(), lr=1e-3
    ).install_hooks(uninterrupted_manager)
    resumed_manager = None
    resumed_optimizer = None

    try:
        _run_moss_logical_step(uninterrupted, uninterrupted_optimizer, _moss_examples(503))
        uninterrupted_manager.commit()
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "model": _cpu_model_state(uninterrupted),
                "optimizer": uninterrupted_optimizer.state_dict(),
            },
            checkpoint_buffer,
        )
        checkpoint_buffer.seek(0)
        checkpoint = torch.load(checkpoint_buffer, map_location="cpu", weights_only=True)

        resumed = _tiny_moss(layers=2)
        resumed.load_state_dict(checkpoint["model"], strict=True)
        resumed_ids = {name: id(parameter) for name, parameter in resumed.named_parameters()}
        place_model(resumed, torch.device("cuda"), 2, block_swap_mode="async_full")
        resumed_manager = resumed.block_swap_manager
        assert isinstance(resumed_manager, TrainableBlockStreamer)
        resumed_optimizer = FusedAdafactor(resumed.named_parameters(), lr=9e-1)
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_optimizer.install_hooks(resumed_manager)
        assert resumed_optimizer.global_step == 1
        assert all(
            value.device.type == "cuda" and value.dtype == torch.float32
            for state in resumed_optimizer.state.values()
            for key, value in state.items()
            if key.startswith("exp_avg_sq")
        )

        examples = _moss_examples(607)
        uninterrupted_outputs = _run_moss_logical_step(
            uninterrupted, uninterrupted_optimizer, examples
        )
        resumed_outputs = _run_moss_logical_step(resumed, resumed_optimizer, examples)
        for actual, expected in zip(resumed_outputs, uninterrupted_outputs):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        uninterrupted_manager.commit()
        resumed_manager.commit()
        _assert_model_states_close(
            _cpu_model_state(resumed), _cpu_model_state(uninterrupted)
        )
        _assert_optimizer_moments_close(resumed_optimizer, uninterrupted_optimizer)
        assert resumed_optimizer.global_step == uninterrupted_optimizer.global_step == 2
        assert {
            name: id(parameter) for name, parameter in resumed.named_parameters()
        } == resumed_ids
        assert resumed_manager.resident_block_indices == ()
        assert all(
            parameter.device.type == "cpu"
            for layer in resumed.layers
            for parameter in layer.parameters()
        )
    finally:
        uninterrupted_optimizer.close()
        if resumed_optimizer is not None:
            resumed_optimizer.close()
        uninterrupted_manager.close()
        if resumed_manager is not None:
            resumed_manager.close()


def test_trainable_streamer_rejects_non_cuda_without_initializing_cuda() -> None:
    model = _Tiny(depth=2)
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    with pytest.raises(ValueError, match="requires a CUDA device"):
        TrainableBlockStreamer(model.layers, torch.device("cpu"))
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_ring_matches_resident_and_updates_every_parameter() -> None:
    torch.manual_seed(51)
    resident = _Tiny().cuda().float()
    streamed = copy.deepcopy(resident).cpu()
    parameter_ids = {name: id(parameter) for name, parameter in streamed.named_parameters()}
    manager = TrainableBlockStreamer(
        streamed.layers, torch.device("cuda"), master_dtype=torch.float32
    )
    streamed.block_swap_manager = manager
    streamed.time.cuda()
    streamed.final_norm.cuda()
    streamed.head.cuda()
    resident_optimizer = FusedAdafactor(resident.named_parameters(), lr=3e-3).install_hooks()
    streamed_optimizer = FusedAdafactor(streamed.named_parameters(), lr=3e-3).install_hooks(manager)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in streamed.named_parameters()}
    value = torch.randn(2, 5, 8, device="cuda", requires_grad=True)
    time = torch.randn(2, 1, device="cuda")

    resident_output = resident(value.detach().clone().requires_grad_(True), time)
    streamed_output = streamed(value.detach().clone().requires_grad_(True), time)
    resident_output.square().mean().backward()
    streamed_output.square().mean().backward()
    resident_optimizer.finish_backward()
    streamed_optimizer.finish_backward()
    manager.finish_backward()

    torch.testing.assert_close(streamed_output, resident_output, rtol=1e-5, atol=1e-6)
    assert streamed_optimizer.updated_parameter_count == len(list(streamed.parameters()))
    assert {name: id(parameter) for name, parameter in streamed.named_parameters()} == parameter_ids
    for (name, actual), (_, expected) in zip(streamed.named_parameters(), resident.named_parameters()):
        assert not torch.equal(actual.cpu(), initial[name]), name
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-6)
    assert manager.d2h_copy_count == len(streamed.layers)
    assert 1 <= manager.peak_resident_blocks <= 2
    assert manager.resident_block_indices == ()
    assert all(parameter.device.type == "cpu" for layer in streamed.layers for parameter in layer.parameters())
    manager.close()
    resident_optimizer.close()
    streamed_optimizer.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dirty_eviction_commit_and_optimizer_resume_preserve_cpu_masters() -> None:
    torch.manual_seed(63)
    model = _Tiny(depth=3).to(dtype=torch.bfloat16)
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    manager = TrainableBlockStreamer(
        model.layers, torch.device("cuda"), master_dtype=torch.bfloat16
    )
    model.block_swap_manager = manager
    model.time.cuda()
    model.final_norm.cuda()
    model.head.cuda()
    optimizer = FusedAdafactor(model.named_parameters(), lr=2e-3).install_hooks(manager)
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    value = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    time = torch.randn(2, 1, device="cuda", dtype=torch.bfloat16)

    model(value, time).float().square().mean().backward()
    optimizer.step()
    manager.finish_backward()
    checkpoint = optimizer.state_dict()
    assert manager.d2h_copy_count == len(model.layers)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert any(not torch.equal(parameter.cpu(), before[name]) for name, parameter in model.named_parameters())

    resumed = FusedAdafactor(model.named_parameters(), lr=9e-1, state_device="cuda")
    resumed.load_state_dict(checkpoint)
    assert resumed.global_step == 1
    assert all(
        value.device.type == "cuda" and value.dtype == torch.float32
        for state in resumed.state.values()
        for key, value in state.items()
        if key.startswith("exp_avg_sq")
    )
    manager.commit()
    exported = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    assert exported.keys() == before.keys()
    manager.close()
    optimizer.close()
    resumed.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_accumulation_backward_stays_clean_until_enabled_boundary() -> None:
    torch.manual_seed(75)
    model = _Tiny(depth=3).float()
    manager = TrainableBlockStreamer(model.layers, "cuda", master_dtype=torch.float32)
    model.block_swap_manager = manager
    model.time.cuda()
    model.final_norm.cuda()
    model.head.cuda()
    optimizer = FusedAdafactor(model.named_parameters(), lr=1e-3).install_hooks(manager)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    first = model(
        torch.randn(2, 3, 8, device="cuda", requires_grad=True),
        torch.randn(2, 1, device="cuda"),
    )
    first.square().mean().backward()
    manager.finish_backward()
    assert manager.d2h_copy_count == 0
    assert all(torch.equal(parameter.cpu(), initial[name]) for name, parameter in model.named_parameters())
    assert all(parameter.grad is not None for parameter in model.parameters())

    optimizer.set_step_enabled(True)
    second = model(
        torch.randn(2, 3, 8, device="cuda", requires_grad=True),
        torch.randn(2, 1, device="cuda"),
    )
    second.square().mean().backward()
    optimizer.finish_backward()
    manager.finish_backward()
    assert manager.d2h_copy_count == len(model.layers)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not torch.equal(parameter.cpu(), initial[name]) for name, parameter in model.named_parameters())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dirty_eviction_rejects_missing_streamed_parameter_updates() -> None:
    model = _Tiny(depth=3).float()
    manager = TrainableBlockStreamer(model.layers, "cuda", master_dtype=torch.float32)
    model.block_swap_manager = manager
    model.time.cuda()
    model.final_norm.cuda()
    model.head.cuda()
    output = model(
        torch.randn(2, 3, 8, device="cuda", requires_grad=True),
        torch.randn(2, 1, device="cuda"),
    )
    with pytest.raises(RuntimeError, match="before all parameter updates"):
        output.square().mean().backward()
    assert manager.requires_reload
    manager.commit()
    assert manager.resident_block_indices == ()

