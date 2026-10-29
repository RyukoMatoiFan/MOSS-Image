"""CUDA validation for pinned host gradients and trainable block streaming."""

import copy

import pytest
import torch
from torch import nn

from training.fused_adafactor import FusedAdafactor
from training.trainable_offload import TrainableBlockStreamer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a reserved CUDA device"),
]


def _logical_step(
    model: nn.Module,
    optimizer: FusedAdafactor,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    optimizer.begin_step()
    for index, values in enumerate(inputs):
        optimizer.set_step_enabled(index == len(inputs) - 1)
        model(values).float().square().mean().backward()
        if index == 0:
            assert optimizer._host_gradients
            assert all(gradient.is_pinned() for gradient in optimizer._host_gradients.values())
            assert all(parameter.grad is None for parameter in model.parameters())
    optimizer.finish_backward()


def _assert_optimizer_state_equal(actual: FusedAdafactor, expected: FusedAdafactor) -> None:
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
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def test_resident_host_accumulation_checkpoint_resume_is_exact() -> None:
    torch.manual_seed(941)
    uninterrupted = nn.Sequential(nn.Linear(6, 9), nn.SiLU(), nn.Linear(9, 4)).cuda()
    optimizer = FusedAdafactor(
        uninterrupted.named_parameters(),
        lr=0.015,
        gradient_accumulation_device="cpu",
    ).install_hooks()
    first_inputs = (
        torch.randn(3, 6, device="cuda"),
        torch.randn(2, 6, device="cuda"),
        torch.randn(4, 6, device="cuda"),
    )
    _logical_step(uninterrupted, optimizer, first_inputs)
    checkpoint_model = {
        name: value.detach().cpu().clone() for name, value in uninterrupted.state_dict().items()
    }
    checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())

    resumed = nn.Sequential(nn.Linear(6, 9), nn.SiLU(), nn.Linear(9, 4))
    resumed.load_state_dict(checkpoint_model)
    resumed.cuda()
    resumed_optimizer = FusedAdafactor(
        resumed.named_parameters(),
        lr=0.9,
        gradient_accumulation_device="cpu",
    )
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    resumed_optimizer.install_hooks()

    second_inputs = (
        torch.randn(5, 6, device="cuda"),
        torch.randn(2, 6, device="cuda"),
    )
    _logical_step(uninterrupted, optimizer, second_inputs)
    _logical_step(resumed, resumed_optimizer, tuple(value.clone() for value in second_inputs))

    for actual, expected in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_optimizer_state_equal(resumed_optimizer, optimizer)
    assert resumed_optimizer.global_step == optimizer.global_step == 2
    optimizer.close()
    resumed_optimizer.close()


def test_explicit_unindexed_cuda_state_device_resolves_to_gradient_device() -> None:
    parameter = nn.Parameter(torch.tensor([0.5, -0.25], device="cuda"))
    optimizer = FusedAdafactor(
        [("weight", parameter)],
        lr=0.01,
        state_device=torch.device("cuda"),
        gradient_accumulation_device="cpu",
    ).install_hooks()

    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    parameter.square().sum().backward()
    optimizer.set_step_enabled(True)
    parameter.square().sum().backward()
    optimizer.finish_backward()

    moment = optimizer.state[parameter]["exp_avg_sq"]
    assert moment.device == parameter.device == torch.device("cuda:0")
    assert moment.dtype == torch.float32
    assert optimizer.global_step == 1
    optimizer.close()


class _SelectiveBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.base = nn.Linear(width, width)
        self.previous_only = nn.Linear(width, width, bias=False)

    def forward(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        cu_seqlens: torch.Tensor,
        use_previous_only: torch.Tensor,
    ) -> torch.Tensor:
        output = value + self.base(value).tanh() * scale
        if bool(use_previous_only.item()):
            output = output + self.previous_only(value) * (cu_seqlens[-1].float() * 0.0 + 0.25)
        return output


class _SelectiveModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_SelectiveBlock(width)])
        self.head = nn.Linear(width, 3)
        self.never_used = nn.Parameter(torch.tensor([0.75]))
        self.block_swap_manager: TrainableBlockStreamer | None = None

    def forward(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        cu_seqlens: torch.Tensor,
        use_previous_only: torch.Tensor,
    ) -> torch.Tensor:
        if self.block_swap_manager is None:
            value = self.layers[0](value, scale, cu_seqlens, use_previous_only)
        else:
            value = self.block_swap_manager.block(
                0,
                value,
                scale,
                cu_seqlens,
                use_previous_only,
                forward_prefetch=None,
                backward_prefetch=None,
                checkpoint_block=True,
            )
        return self.head(value)


def test_streamed_host_accumulation_updates_previous_only_and_ignores_never_used() -> None:
    torch.manual_seed(977)
    resident = _SelectiveModel().cuda()
    streamed = copy.deepcopy(resident)
    manager = TrainableBlockStreamer(streamed.layers, "cuda", master_dtype=torch.float32)
    assert len(manager.owned_cuda_buffers()) == 2
    assert all(buffer.device.type == "cuda" for buffer in manager.owned_cuda_buffers())
    streamed.block_swap_manager = manager
    reference_parameters = [
        (name, parameter)
        for name, parameter in resident.named_parameters()
        if name != "never_used"
    ]
    reference = FusedAdafactor(reference_parameters, lr=0.01)
    host = FusedAdafactor(
        streamed.named_parameters(),
        lr=0.01,
        gradient_accumulation_device="cpu",
    ).install_hooks(manager)
    never_used_initial = streamed.never_used.detach().clone()
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32, device="cuda")
    batches = (
        (
            torch.randn(1, 7, 8, device="cuda"),
            torch.randn(1, 1, 8, device="cuda"),
            cu_seqlens,
            torch.ones((), device="cuda"),
        ),
        (
            torch.randn(1, 7, 8, device="cuda"),
            torch.randn(1, 1, 8, device="cuda"),
            cu_seqlens,
            torch.zeros((), device="cuda"),
        ),
    )

    host.begin_step()
    reference.begin_step()
    for index, batch in enumerate(batches):
        last = index == len(batches) - 1
        host.set_step_enabled(last)
        streamed(*batch).float().square().mean().backward()
        resident(*tuple(value.clone() for value in batch)).float().square().mean().backward()
        manager.finish_backward()
        if not last:
            assert host._host_gradients
            assert all(gradient.is_pinned() for gradient in host._host_gradients.values())
            assert all(parameter.grad is None for parameter in streamed.parameters())
            assert manager.d2h_copy_count == 0
    host.finish_backward()
    reference.step()

    streamed_state = dict(streamed.named_parameters())
    resident_state = dict(resident.named_parameters())
    for name, actual in streamed_state.items():
        torch.testing.assert_close(actual.cpu(), resident_state[name].cpu(), rtol=1e-6, atol=1e-7)
    previous_only = streamed.layers[0].previous_only.weight
    assert host.state[previous_only]["step"] == 1
    assert host.state[streamed.never_used] == {}
    assert torch.equal(streamed.never_used, never_used_initial)
    assert host.updated_parameter_count == len(list(streamed.parameters())) - 1
    assert manager.d2h_copy_count == 1
    assert manager.resident_block_indices == ()
    manager.close()
    host.close()
    reference.close()
