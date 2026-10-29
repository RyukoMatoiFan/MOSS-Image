"""CPU-only coverage for FP32 host gradient accumulation."""

import copy
import math

import pytest
import torch

from training.fused_adafactor import FusedAdafactor


@pytest.fixture(autouse=True)
def _forbid_cuda_initialization(monkeypatch: pytest.MonkeyPatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    def fail_cuda_init() -> None:
        raise AssertionError("CPU host-gradient tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "init", fail_cuda_init)
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail_cuda_init)
    yield
    assert not torch.cuda.is_initialized()


def test_host_accumulation_releases_early_grads_and_matches_device_reference() -> None:
    torch.manual_seed(104)
    host_model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.SiLU(), torch.nn.Linear(7, 3))
    reference_model = copy.deepcopy(host_model)
    host = FusedAdafactor(
        host_model.named_parameters(),
        lr=0.02,
        gradient_accumulation_device="cpu",
    ).install_hooks()
    reference = FusedAdafactor(reference_model.named_parameters(), lr=0.02)
    initial = {name: parameter.detach().clone() for name, parameter in host_model.named_parameters()}
    batches = [torch.randn(4, 5), torch.randn(2, 5), torch.randn(6, 5)]

    host.begin_step()
    host.set_step_enabled(False)
    for batch in batches[:-1]:
        host_model(batch).square().sum().mul(1 / 37).backward()

    assert all(parameter.grad is None for parameter in host_model.parameters())
    assert all(torch.equal(parameter, initial[name]) for name, parameter in host_model.named_parameters())
    assert len(host._host_gradients) == len(list(host_model.parameters()))
    assert all(value.device.type == "cpu" and value.dtype == torch.float32
               for value in host._host_gradients.values())

    host.set_step_enabled(True)
    host_model(batches[-1]).square().sum().mul(1 / 37).backward()
    host_norm = host.finish_backward()

    reference.begin_step()
    for batch in batches:
        reference_model(batch).square().sum().mul(1 / 37).backward()
    expected_norm = math.sqrt(
        sum(float(parameter.grad.float().square().sum()) for parameter in reference_model.parameters())
    )
    reference.step()

    assert host.global_step == reference.global_step == 1
    assert host_norm == pytest.approx(expected_norm)
    assert host._host_gradients == {}
    assert all(parameter.grad is None for parameter in host_model.parameters())
    for actual, expected in zip(host_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_host_only_gradient_is_updated_and_never_used_parameter_is_ignored() -> None:
    early = torch.nn.Parameter(torch.tensor([0.4, -0.7]))
    final = torch.nn.Parameter(torch.tensor([-0.2, 0.9]))
    unused = torch.nn.Parameter(torch.tensor([1.5, -1.1]))
    optimizer = FusedAdafactor(
        [("early", early), ("final", final), ("unused", unused)],
        lr=0.03,
        gradient_accumulation_device="cpu",
    ).install_hooks()
    initial = {"early": early.detach().clone(), "final": final.detach().clone(),
               "unused": unused.detach().clone()}

    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    (early.square().sum() * 0.4).backward()
    assert early.grad is None

    optimizer.set_step_enabled(True)
    (final.square().sum() * 0.6).backward()
    optimizer.finish_backward()

    assert optimizer.updated_parameter_count == 2
    assert not torch.equal(early, initial["early"])
    assert not torch.equal(final, initial["final"])
    assert torch.equal(unused, initial["unused"])
    assert optimizer.state[unused] == {}


def test_pending_host_gradients_are_not_silently_checkpointed() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
    optimizer = FusedAdafactor(
        [("weight", parameter)],
        lr=0.01,
        gradient_accumulation_device="cpu",
    ).install_hooks()
    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    parameter.square().sum().backward()

    with pytest.raises(RuntimeError, match="uncommitted optimizer step"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="active step"):
        optimizer.set_gradient_accumulation_device("device")

    optimizer.set_step_enabled(True)
    optimizer.finish_backward()
    checkpoint = optimizer.state_dict()
    assert checkpoint["fused_adafactor"]["global_step"] == 1
    assert all("host" not in key for state in checkpoint["state"].values() for key in state)


def test_nonfinite_host_gradient_latches_failure_and_checkpoint_reload_recovers() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
    initial = parameter.detach().clone()
    optimizer = FusedAdafactor(
        [("weight", parameter)],
        lr=0.01,
        gradient_accumulation_device="cpu",
    ).install_hooks()
    committed = copy.deepcopy(optimizer.state_dict())

    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    parameter.grad = torch.tensor([float("nan"), 1.0])
    optimizer._stash_gradient_on_host(parameter)
    optimizer.set_step_enabled(True)
    with pytest.raises(FloatingPointError, match="requires checkpoint reload"):
        optimizer.finish_backward()
    assert optimizer.requires_reload
    with pytest.raises(RuntimeError, match="failed partial step"):
        optimizer.state_dict()

    parameter.data.copy_(initial)
    optimizer.load_state_dict(committed)
    assert not optimizer.requires_reload
    optimizer.begin_step()
    optimizer.set_step_enabled(False)
    (parameter.square().sum() * 0.3).backward()
    optimizer.set_step_enabled(True)
    optimizer.finish_backward()
    assert optimizer.global_step == 1
    assert torch.isfinite(parameter).all()


def test_per_parameter_weight_decay_and_decay_to_one() -> None:
    plain = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    toward_one = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    excluded = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = FusedAdafactor(
        [("plain", plain), ("toward_one", toward_one), ("excluded", excluded)],
        lr=0.1,
        weight_decay=0.2,
        parameter_options={
            "toward_one": {"decay_to_one": True},
            "excluded": {"weight_decay": 0.0},
        },
    )
    for parameter in (plain, toward_one, excluded):
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    torch.testing.assert_close(plain, torch.tensor([1.96, -0.98]), rtol=0, atol=1e-7)
    torch.testing.assert_close(toward_one, torch.tensor([1.98, -0.96]), rtol=0, atol=1e-7)
    torch.testing.assert_close(excluded, torch.tensor([2.0, -1.0]), rtol=0, atol=0)


def test_cuda_alias_preserves_legacy_device_accumulation_mode() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = FusedAdafactor(
        [("weight", parameter)],
        lr=0.01,
        gradient_accumulation_device="cuda",
    )
    assert optimizer.gradient_accumulation_device == "device"
    with pytest.raises(ValueError, match="must be 'device', 'cuda', or 'cpu'"):
        optimizer.set_gradient_accumulation_device("disk")


def test_streamer_is_told_that_host_mode_allows_unused_parameters() -> None:
    class Streamer:
        def __init__(self) -> None:
            self.require_all_updates = True
            self.step_enabled = True

        def set_require_all_updates(self, required: bool) -> None:
            self.require_all_updates = required

        def set_step_enabled(self, enabled: bool) -> None:
            self.step_enabled = enabled

        def mark_updated(self, _parameter: torch.nn.Parameter) -> None:
            pass

        def mark_failed(self, _parameter: torch.nn.Parameter) -> None:
            pass

    parameter = torch.nn.Parameter(torch.ones(2))
    streamer = Streamer()
    optimizer = FusedAdafactor(
        [("weight", parameter)],
        lr=0.01,
        gradient_accumulation_device="cpu",
    ).install_hooks(streamer)
    assert not streamer.require_all_updates
    optimizer.set_gradient_accumulation_device("device")
    assert streamer.require_all_updates
