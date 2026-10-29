"""Checks for the flow-matching target and the report's sigma distribution."""

from __future__ import annotations

import json
import math

import torch
from safetensors.torch import save_file
from torch import nn

from training.config import TrainConfig, load_config
from training.flow import FlowMatcher, discrete_sigma_probabilities, flow_cdf
from training.train import (
    _group_loss,
    _group_squared_error,
    _load_rows,
    _record_config,
    _require_budget,
    _scale_step_error,
)


def test_flow_cdf_matches_an_independent_normal_probability() -> None:
    """F(0.4) equals 1 - Phi((logit(0.6) - mu) / s), computed with math.erf."""

    mu, width, sigma = -1.09, 0.95, 0.4
    logit = math.log((1 - sigma) / sigma)
    expected = 1 - 0.5 * (1 + math.erf((logit - mu) / (width * math.sqrt(2))))
    actual = flow_cdf(torch.tensor(sigma), mu, width).item()
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)


def test_discrete_probabilities_sum_to_one_and_cover_the_grid() -> None:
    """Equation (4) is a distribution on sigma = j / 1000, j = 1..1000."""

    probabilities = discrete_sigma_probabilities()
    assert probabilities.shape == (1000,)
    assert torch.all(probabilities > 0)
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-8)
    matcher = FlowMatcher()
    drawn = matcher.sample_sigma(8, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert torch.all((drawn * 1000).round() == drawn * 1000)
    assert torch.all((drawn > 0) & (drawn <= 1))


def test_euler_step_with_the_true_velocity_reaches_the_lower_sigma() -> None:
    """z + (sigma_next - sigma) * (eps - z0) equals the interpolant at sigma_next."""

    matcher = FlowMatcher()
    clean = torch.tensor([[1.0, -2.0, 0.5]])
    noise = torch.tensor([[0.25, 0.5, -1.0]])
    sigma = torch.tensor([0.4])
    noisy, velocity = matcher.noisy_latent(clean, sigma, noise)
    sigma_next = 0.1
    stepped = noisy + (sigma_next - sigma.view(1, 1)) * velocity
    expected = sigma_next * noise + (1 - sigma_next) * clean
    assert torch.allclose(stepped, expected, atol=1e-6)
    assert torch.allclose(matcher.timestep(sigma), torch.tensor([400.0]))


class _ZeroHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, timestep, context, mask=None, ref_hidden_states=None):
        return torch.zeros_like(hidden)


class _FixedSigma(FlowMatcher):
    def sample_sigma(self, batch, device, generator=None):
        return torch.full((batch,), 0.5, device=device)


def test_group_loss_weights_shapes_by_token_count() -> None:
    """Two shapes share one mean over elements, not one mean per image."""

    prepared = [
        (torch.zeros(4, 2, 2), torch.zeros(1, 4), []),
        (torch.ones(4, 1, 1), torch.zeros(1, 4), []),
    ]
    torch.manual_seed(5)
    loss = _group_loss(_ZeroHead(), _FixedSigma(total=8), prepared, torch.device("cpu"))
    torch.manual_seed(5)
    noise_large = torch.randn(1, 4, 2, 2)
    noise_small = torch.randn(1, 4, 1, 1)
    large = noise_large.square()
    small = (noise_small - 1).square()
    expected = torch.cat((large.flatten(), small.flatten())).mean()
    per_image = 0.5 * (large.mean() + small.mean())
    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, per_image)


def test_accumulated_microbatches_use_the_joint_element_count() -> None:
    """Two micro-batches make the same mean as one batch of both images."""

    big = [(torch.zeros(4, 2, 2), torch.zeros(1, 4), [])]
    small = [(torch.ones(4, 1, 1), torch.zeros(1, 4), [])]
    torch.manual_seed(5)
    state = torch.random.get_rng_state()
    big_error, big_count = _group_squared_error(_ZeroHead(), _FixedSigma(total=8), big, torch.device("cpu"))
    small_error, small_count = _group_squared_error(_ZeroHead(), _FixedSigma(total=8), small, torch.device("cpu"))
    combined = _scale_step_error(big_error, big_count + small_count)
    combined = combined + _scale_step_error(small_error, big_count + small_count)
    torch.random.set_rng_state(state)
    joint = _group_loss(_ZeroHead(), _FixedSigma(total=8), big + small, torch.device("cpu"))
    per_batch = 0.5 * (big_error / big_count + small_error / small_count)
    assert torch.allclose(combined, joint)
    assert not torch.allclose(combined, per_batch)
    left = torch.tensor(float(big_error), requires_grad=True)
    right = torch.tensor(float(small_error), requires_grad=True)
    total = big_count + small_count
    (_scale_step_error(left, total) + _scale_step_error(right, total)).backward()
    assert torch.allclose(left.grad, torch.tensor(1.0 / total))
    assert torch.allclose(right.grad, torch.tensor(1.0 / total))


def test_budget_rejects_a_latent_grid_that_disagrees_with_the_pixels() -> None:
    """Pixel height and width have to be the latent grid times 16."""

    try:
        _require_budget(
            [{"file": "x.safetensors", "height": 64, "width": 64, "latent": [128, 200, 200], "refs": []}],
            1024,
        )
    except ValueError as exc:
        assert "200" in str(exc)
    else:
        raise AssertionError("a latent grid other than height/16 should fail")
    _require_budget(
        [{"file": "x.safetensors", "height": 64, "width": 64, "latent": [128, 4, 4], "refs": []}],
        128,
    )


def test_resolution_rejects_an_oversized_cache_and_is_recorded(tmp_path) -> None:
    """The pixel budget is enforced, and the run writes it down."""

    cache = tmp_path / "cache"
    cache.mkdir()
    row = {"file": "000000.safetensors", "height": 2048, "width": 2048, "refs": [[128, 8, 8]]}
    (cache / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    save_file({"context": torch.zeros(1, 4)}, str(cache / "empty.safetensors"))
    try:
        _load_rows(TrainConfig(cache_dir=str(cache), resolution=1024))
    except ValueError as exc:
        assert "2048" in str(exc)
    else:
        raise AssertionError("an image above the budget should fail")
    fitting = tmp_path / "fit"
    fitting.mkdir()
    small = {"file": "000000.safetensors", "height": 64, "width": 64, "refs": [[128, 2, 2]]}
    (fitting / "index.jsonl").write_text(json.dumps(small) + "\n", encoding="utf-8")
    save_file({"context": torch.zeros(1, 4)}, str(fitting / "empty.safetensors"))
    rows, eval_rows, _ = _load_rows(TrainConfig(cache_dir=str(fitting), resolution=128))
    assert len(rows) == 1 and eval_rows == []
    config_path = tmp_path / "run.yaml"
    config_path.write_text("resolution: 8\nmode: lora\n", encoding="utf-8")
    try:
        load_config(config_path)
    except ValueError as exc:
        assert "resolution" in str(exc)
    else:
        raise AssertionError("resolution below 16 should fail")
    recorded = tmp_path / "run"
    _record_config(TrainConfig(resolution=768, cache_dir=str(fitting)), recorded)
    payload = json.loads((recorded / "train_config.json").read_text(encoding="utf-8"))
    assert payload["resolution"] == 768
