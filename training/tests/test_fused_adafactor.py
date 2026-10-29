"""Manual-LR math, accumulation hooks, and durable optimizer-state checks."""

import copy
import math

import pytest
import torch

from training.fused_adafactor import FusedAdafactor


def _reference_first_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    *,
    lr: float,
    eps: float,
    clip_threshold: float,
    weight_decay: float,
) -> torch.Tensor:
    squared = gradient.square() + eps
    row = squared.mean(dim=-1)
    column = squared.mean(dim=-2)
    row_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
    column_factor = column.rsqrt().unsqueeze(-2)
    update = gradient * row_factor * column_factor
    update = update / max(1.0, float(update.square().mean().sqrt()) / clip_threshold)
    return parameter * (1.0 - lr * weight_decay) - lr * update


def test_manual_lr_factored_math_has_fp32_second_moments_only() -> None:
    initial = torch.tensor([[0.5, -0.2, 0.9], [1.2, -0.7, 0.1]], dtype=torch.float32)
    gradient = torch.tensor([[0.3, -1.1, 0.7], [-0.4, 0.2, 1.3]], dtype=torch.float32)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = FusedAdafactor(
        [("matrix", parameter)],
        lr=0.03,
        eps=1e-12,
        clip_threshold=0.6,
        weight_decay=0.07,
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    expected = _reference_first_step(
        initial,
        gradient,
        lr=0.03,
        eps=1e-12,
        clip_threshold=0.6,
        weight_decay=0.07,
    )
    torch.testing.assert_close(parameter, expected, rtol=2e-6, atol=2e-7)
    state = optimizer.state[parameter]
    assert state["exp_avg_sq_row"].shape == (2,)
    assert state["exp_avg_sq_col"].shape == (3,)
    assert state["exp_avg_sq_row"].dtype == state["exp_avg_sq_col"].dtype == torch.float32
    assert "exp_avg" not in state
    assert state["step"] == optimizer.global_step == 1
    assert optimizer.gradient_norm == pytest.approx(float(torch.linalg.vector_norm(gradient)))


def test_post_accumulate_hooks_update_and_clear_only_at_boundary() -> None:
    torch.manual_seed(18)
    fused_model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.SiLU(), torch.nn.Linear(7, 3))
    reference_model = copy.deepcopy(fused_model)
    fused = FusedAdafactor(fused_model.named_parameters(), lr=0.02).install_hooks()
    reference = FusedAdafactor(reference_model.named_parameters(), lr=0.02)
    initial = {name: value.detach().clone() for name, value in fused_model.named_parameters()}
    batches = [torch.randn(4, 5), torch.randn(6, 5)]

    fused.begin_step()
    fused.set_step_enabled(False)
    fused_model(batches[0]).square().mean().backward()
    assert all(torch.equal(parameter, initial[name]) for name, parameter in fused_model.named_parameters())
    assert all(parameter.grad is not None for parameter in fused_model.parameters())

    fused.set_last_group_step_boundary(True)
    fused_model(batches[1]).square().mean().backward()
    fused.finish_backward()
    fused.step()  # Calling both completion APIs must not increment twice.

    reference.begin_step()
    reference_model(batches[0]).square().mean().backward()
    reference_model(batches[1]).square().mean().backward()
    expected_norm = math.sqrt(sum(float(parameter.grad.square().sum()) for parameter in reference_model.parameters()))
    reference.step()

    assert fused.global_step == reference.global_step == 1
    assert fused.updated_parameter_count == len(list(fused_model.parameters()))
    assert fused.gradient_norm == pytest.approx(expected_norm)
    assert all(parameter.grad is None for parameter in fused_model.parameters())
    for actual, expected in zip(fused_model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fused.close()


def test_nonfinite_gradient_latches_partial_step_and_blocks_save() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = FusedAdafactor([("vector", parameter)], lr=0.1)
    parameter.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(FloatingPointError, match="requires checkpoint reload"):
        optimizer.step()
    assert not torch.isfinite(parameter).all()
    assert optimizer.global_step == 0
    with pytest.raises(RuntimeError, match="failed partial step"):
        optimizer.state_dict()


def test_state_resume_validates_names_and_shapes_and_continues_exactly() -> None:
    torch.manual_seed(27)
    parameters = {
        "matrix": torch.nn.Parameter(torch.randn(4, 9)),
        "vector": torch.nn.Parameter(torch.randn(9)),
    }
    optimizer = FusedAdafactor(parameters.items(), lr=0.04, weight_decay=0.02)
    for parameter in parameters.values():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())

    restored_parameters = {
        name: torch.nn.Parameter(parameter.detach().clone()) for name, parameter in parameters.items()
    }
    restored = FusedAdafactor(restored_parameters.items(), lr=0.9)
    restored.load_state_dict(checkpoint)
    assert restored.global_step == 1
    for state in restored.state.values():
        assert all(value.dtype == torch.float32 for key, value in state.items() if key.startswith("exp_avg_sq"))

    for parameter, resumed_parameter in zip(parameters.values(), restored_parameters.values()):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient
        resumed_parameter.grad = gradient.clone()
    optimizer.step()
    restored.step()
    for parameter, resumed_parameter in zip(parameters.values(), restored_parameters.values()):
        torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)

    wrong_name = FusedAdafactor([("renamed", torch.nn.Parameter(torch.randn(4, 9))),
                                 ("vector", torch.nn.Parameter(torch.randn(9)))], lr=0.04)
    with pytest.raises(ValueError, match="names/order"):
        wrong_name.load_state_dict(checkpoint)
    wrong_shape = FusedAdafactor([("matrix", torch.nn.Parameter(torch.randn(5, 9))),
                                  ("vector", torch.nn.Parameter(torch.randn(9)))], lr=0.04)
    with pytest.raises(ValueError, match="shapes"):
        wrong_shape.load_state_dict(checkpoint)


def test_bf16_parameter_resume_does_not_quantize_fp32_moments() -> None:
    parameter = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.bfloat16))
    optimizer = FusedAdafactor([("weight", parameter)], lr=0.01)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    expected = {
        key: value.clone()
        for key, value in checkpoint["state"][0].items()
        if isinstance(value, torch.Tensor)
    }

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = FusedAdafactor([("weight", restored_parameter)], lr=0.01)
    restored.load_state_dict(checkpoint)
    for key, value in expected.items():
        assert torch.equal(restored.state[restored_parameter][key], value)
        assert restored.state[restored_parameter][key].dtype == torch.float32


def test_missing_parameter_state_stays_empty_and_initializes_lazily_after_resume() -> None:
    used = torch.nn.Parameter(torch.randn(3, 4))
    dormant = torch.nn.Parameter(torch.randn(4), requires_grad=False)
    optimizer = FusedAdafactor([("used", used), ("dormant", dormant)], lr=0.01)
    used.grad = torch.randn_like(used)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    assert checkpoint["state"].get(1, {}) == {}

    restored_used = torch.nn.Parameter(used.detach().clone())
    restored_dormant = torch.nn.Parameter(dormant.detach().clone(), requires_grad=False)
    restored = FusedAdafactor(
        [("used", restored_used), ("dormant", restored_dormant)], lr=0.01
    )
    restored.load_state_dict(checkpoint)
    assert restored.state[restored_dormant] == {}
    restored_used.grad = torch.randn_like(restored_used)
    restored_dormant.grad = torch.randn_like(restored_dormant)
    restored.step()
    assert "exp_avg_sq" in restored.state[restored_dormant]


def test_resume_rejects_optimizer_kind_and_parameter_precision_changes() -> None:
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    optimizer = FusedAdafactor([("weight", parameter)], lr=0.01)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    checkpoint = optimizer.state_dict()

    wrong_kind = copy.deepcopy(checkpoint)
    wrong_kind["fused_adafactor"]["optimizer_kind"] = "other"
    with pytest.raises(ValueError, match="optimizer kind"):
        FusedAdafactor([("weight", torch.nn.Parameter(parameter.detach().clone()))], lr=0.01).load_state_dict(
            wrong_kind
        )
    bf16 = FusedAdafactor(
        [("weight", torch.nn.Parameter(parameter.detach().to(torch.bfloat16)))], lr=0.01
    )
    with pytest.raises(ValueError, match="master dtype"):
        bf16.load_state_dict(checkpoint)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_cuda_state_stays_fp32_on_gpu_across_cpu_park_and_resume() -> None:
    torch.manual_seed(39)
    parameter = torch.nn.Parameter(torch.randn(8, 16, device="cuda", dtype=torch.bfloat16))
    optimizer = FusedAdafactor([("weight", parameter)], lr=1e-2).install_hooks()
    parameter.square().float().mean().backward()
    optimizer.finish_backward()
    saved_parameter = parameter.detach().cpu().clone()
    checkpoint = optimizer.state_dict()
    assert all(
        value.device.type == "cuda" and value.dtype == torch.float32
        for key, value in optimizer.state[parameter].items()
        if key.startswith("exp_avg_sq")
    )

    restored_parameter = torch.nn.Parameter(saved_parameter.clone())
    restored = FusedAdafactor(
        [("weight", restored_parameter)], lr=1e-2, state_device=torch.device("cuda")
    )
    restored.load_state_dict(checkpoint)
    assert all(
        value.device.type == "cuda" and value.dtype == torch.float32
        for key, value in restored.state[restored_parameter].items()
        if key.startswith("exp_avg_sq")
    )

