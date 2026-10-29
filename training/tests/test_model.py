"""RoPE, edit packing, and checkpoint round-trip for the DiT."""

from __future__ import annotations

import torch

from training.model import MOSSImageTransformer, RoPE, rope_coordinates, rope_phases, rotate_rope
from training.weights import load_transformer, save_transformer


def _tiny() -> MOSSImageTransformer:
    return MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=2, text_dim=16, freq_dim=8, latent_channels=4,
    )


def test_text_to_image_coordinates_match_the_released_formula() -> None:
    """Target tokens use r=0, t=text length, and centered spatial axes."""

    height, width, text_length = 4, 6, 3
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = torch.stack((torch.zeros_like(rows), torch.full_like(rows, text_length), rows, cols), -1)
    image = image.flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4)
    text[:, 1] = torch.arange(text_length).float()
    expected = torch.cat((image, text), 0)
    actual = rope_coordinates([(0, height, width)], text_length, torch.device("cpu"))
    assert torch.equal(actual, expected)


def test_reference_axis_keeps_target_and_text_coordinates() -> None:
    """References take r=1 after the target grid and do not move the text tail."""

    base = rope_coordinates([(0, 4, 4)], 3, torch.device("cpu"))
    edit = rope_coordinates([(0, 4, 4), (1, 2, 2)], 3, torch.device("cpu"))
    assert torch.equal(edit[:16], base[:16])
    assert torch.equal(edit[-3:], base[-3:])
    assert torch.all(edit[16:20, 0] == 1)


def test_zero_reference_index_leaves_its_feature_pair_unrotated() -> None:
    """With r=0 the 16 r-bands are an identity rotation, which is 32 of 128 dims."""

    bands = RoPE(128, 10000.0).bands
    coordinates = rope_coordinates([(0, 4, 4)], 3, torch.device("cpu"))
    rope = rope_phases(coordinates, bands)
    tokens = torch.randn(1, coordinates.shape[0], 2, 128)
    rotated = rotate_rope(tokens, rope)
    assert torch.allclose(rotated[..., 0:16], tokens[..., 0:16], atol=1e-5)
    assert torch.allclose(rotated[..., 64:80], tokens[..., 64:80], atol=1e-5)
    assert not torch.allclose(rotated[..., 16:32], tokens[..., 16:32])


def test_edit_forward_predicts_the_target_grid_only() -> None:
    """A reference changes the velocity and does not change its shape."""

    torch.manual_seed(0)
    model = _tiny().eval()
    target = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 3, 16)
    timestep = torch.tensor([500.0])
    reference = torch.randn(1, 4, 2, 6)
    plain = model(target, timestep, context)
    edited = model(target, timestep, context, ref_hidden_states=[reference])
    empty = model(target, timestep, context, ref_hidden_states=[])
    assert plain.shape == edited.shape == target.shape
    assert torch.allclose(plain, empty)
    assert not torch.allclose(plain, edited)


def test_skip_layers_removes_those_blocks() -> None:
    """Skipping every joint block is a different function from running them."""

    torch.manual_seed(1)
    model = _tiny().eval()
    target = torch.randn(1, 4, 4, 4)
    context = torch.randn(1, 2, 16)
    full = model(target, torch.tensor(250.0), context)
    skipped = model(target, torch.tensor(250.0), context, skip_layers=[0, 1])
    assert not torch.allclose(full, skipped)
    try:
        model(target, torch.tensor(250.0), context, skip_layers=[9])
    except ValueError:
        return
    raise AssertionError("an out-of-range skip index should fail")


def test_transformer_round_trip(tmp_path) -> None:
    """A saved state dict reloads to the same weights and ignores stale RoPE bands."""

    torch.manual_seed(2)
    model = _tiny()
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    save_transformer(model, path)
    loaded = load_transformer(tmp_path)
    for key, value in model.state_dict().items():
        if key == "rope.bands":
            continue
        assert torch.equal(value, loaded.state_dict()[key])
    assert loaded.rope.bands.dtype == torch.float32
