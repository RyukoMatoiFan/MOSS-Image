"""Adafactor LR semantics, factored storage, and mixed-master resume checks."""

import copy

import pytest
import torch

from training.optimizers import build_adafactor


def test_adafactor_uses_native_relative_learning_rate_and_decay():
    parameter = torch.nn.Parameter(torch.full((8, 16), 2.0))
    optimizer = build_adafactor([parameter], lr=0.01, weight_decay=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    # First-step unit normalized gradient, parameter RMS 2, and decay 0.002.
    torch.testing.assert_close(parameter, torch.full_like(parameter, 1.978))
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["foreach"] is False


def test_adafactor_matrix_state_is_factored():
    parameter = torch.nn.Parameter(torch.randn(128, 512))
    optimizer = build_adafactor([parameter], lr=0.01, weight_decay=0.0)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["row_var"].shape == (128, 1)
    assert state["col_var"].shape == (1, 512)
    assert "exp_avg" not in state and "variance" not in state
    state_elements = sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor))
    assert state_elements == 128 + 512 + 1
    assert state_elements < parameter.numel() // 50


@pytest.mark.parametrize("mixed_cuda", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_adafactor_master_updates_and_state_reload_parity(mixed_cuda):
    if mixed_cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA for mixed CPU/CUDA optimizer masters")
    torch.manual_seed(37)
    devices = ["cpu", "cuda" if mixed_cuda else "cpu", "cpu"]
    shapes = [(7, 19), (11, 5), (13,)]
    parameters = [torch.nn.Parameter(torch.randn(shape, device=device))
                  for shape, device in zip(shapes, devices)]
    optimizer = build_adafactor(parameters, lr=0.02, weight_decay=0.01)
    initial = [parameter.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert all(not torch.equal(before, after) for before, after in zip(initial, parameters))
    resumed_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    resumed = build_adafactor(resumed_parameters, lr=0.9, weight_decay=0.0)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for step in range(3):
        for parameter, restored in zip(parameters, resumed_parameters):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient
            restored.grad = gradient.clone()
        optimizer.step()
        resumed.step()
        for parameter, restored in zip(parameters, resumed_parameters):
            torch.testing.assert_close(parameter, restored, rtol=0, atol=0)
            for key, value in optimizer.state[parameter].items():
                torch.testing.assert_close(value, resumed.state[restored][key], rtol=0, atol=0)
                if key != "step":
                    assert value.device == parameter.device
