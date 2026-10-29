"""Comfy storage decoding, including embedding rows and malformed metadata."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from training.comfy_weights import decode_comfy_state, load_comfy_state
from training.convrot import rotate_hadamard


def marker(**overrides):
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
    config.update(overrides)
    return torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)


def fixture_state():
    return {"layer.weight": torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.int8),
            "layer.weight_scale": torch.tensor([[0.25], [0.125]], dtype=torch.float32),
            "layer.comfy_quant": marker(), "layer.bias": torch.tensor([1., 2.]),
            "vae.num_batches_tracked": torch.tensor(9)}


def test_decode_linear_matches_explicit_regular_hadamard_and_embedding():
    state = fixture_state()
    dense = decode_comfy_state(state, torch.float32)
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]]) / 2
    rotated_weight = state["layer.weight"].float() * state["layer.weight_scale"]
    torch.testing.assert_close(dense["layer.weight"], rotated_weight @ h4)
    x = torch.randn(3, 4)
    torch.testing.assert_close(F.linear(x, dense["layer.weight"]),
                               F.linear(x @ h4, rotated_weight))
    ids = torch.tensor([1, 0, 1])
    torch.testing.assert_close(F.embedding(ids, dense["layer.weight"]), (rotated_weight @ h4)[ids])
    assert dense["vae.num_batches_tracked"].dtype == torch.int64
    assert len(state) == 5  # Caller mapping survives decoding.
    assert set(dense) == {"layer.weight", "layer.bias", "vae.num_batches_tracked"}


def test_group256_and_file_load_dense_passthrough(tmp_path):
    q = torch.randint(-128, 128, (5, 256), dtype=torch.int8)
    scale = torch.full((5, 1), 0.01)
    state = {"embed.weight": q, "embed.weight_scale": scale,
             "embed.comfy_quant": marker(convrot_groupsize=256),
             "norm.weight": torch.ones(5)}
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    result = load_comfy_state(path)
    assert result["embed.weight"].dtype == torch.bfloat16
    assert result["norm.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(result["embed.weight"], rotate_hadamard(q.float() * scale, 256).bfloat16())


def test_nonrotated_int8():
    state = fixture_state()
    state["layer.comfy_quant"] = torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    result = decode_comfy_state(state, torch.float32)
    torch.testing.assert_close(result["layer.weight"], state["layer.weight"].float() * state["layer.weight_scale"])


@pytest.mark.parametrize("mutation", [
    lambda s: s.pop("layer.weight_scale"),
    lambda s: s.pop("layer.comfy_quant"),
    lambda s: s.update({"layer.comfy_quant": marker(format="gptq")}),
    lambda s: s.update({"layer.comfy_quant": torch.tensor([255], dtype=torch.uint8)}),
    lambda s: s.update({"layer.weight_scale": torch.ones(2)}),
    lambda s: s.update({"layer.weight_scale": torch.zeros(2, 1)}),
    lambda s: s.update({"layer.weight_scale": torch.full((2, 1), float("nan"))}),
    lambda s: s.update({"layer.weight": torch.ones(2, 4, dtype=torch.uint8)}),
    lambda s: s.update({"layer.comfy_quant": marker(convrot_groupsize=16)}),
    lambda s: s.update({"layer.comfy_quant": marker(convrot_groupsize=8)}),
    lambda s: s.update({"layer.comfy_quant": marker(convrot="yes")}),
])
def test_malformed_or_unsupported_storage_rejected(mutation):
    state = fixture_state()
    mutation(state)
    with pytest.raises(ValueError):
        decode_comfy_state(state)
