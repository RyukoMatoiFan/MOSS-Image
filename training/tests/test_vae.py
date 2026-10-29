"""Strict native and Comfy VAE conversion with complete state/output parity."""

import pytest
import torch
from safetensors.torch import save_file

from training.vae import _original_key, load_vae


@pytest.fixture
def tiny_vae(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.AutoencoderKLFlux2(
        block_out_channels=(8, 16), decoder_block_out_channels=(8, 16),
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        norm_num_groups=4, latent_channels=4, layers_per_block=1,
    ).eval()
    model.bn.running_mean.normal_()
    model.bn.running_var.uniform_(0.5, 1.5)
    model.bn.num_batches_tracked.fill_(19)
    model.save_config(tmp_path)
    return model, tmp_path


def _comfy_state(model):
    result = {}
    for key, value in model.state_dict().items():
        destination = _original_key(key, len(model.decoder.up_blocks))
        if ".attn_1." in destination and value.ndim == 2:
            value = value[:, :, None, None]
        result[destination] = value.contiguous()
    return result


@pytest.mark.parametrize("comfy", [False, True])
def test_single_file_vae_state_and_encode_decode_parity(tiny_vae, comfy):
    model, directory = tiny_vae
    state = _comfy_state(model) if comfy else model.state_dict()
    path = directory / "vae.safetensors"
    save_file(state, str(path))
    restored = load_vae(directory / "config.json", "cpu", path)
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert restored.bn.num_batches_tracked.dtype == torch.int64
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = model.encode(pixels).latent_dist.mode()
        actual = restored.encode(pixels).latent_dist.mode()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.decode(actual).sample, model.decode(expected).sample, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape"])
def test_single_file_vae_rejects_incomplete_or_wrong_state(tiny_vae, failure):
    model, directory = tiny_vae
    state = _comfy_state(model)
    if failure == "missing":
        del state["bn.running_mean"]
    elif failure == "extra":
        state["unmanaged_constant"] = torch.ones(1)
    else:
        state["encoder.conv_in.weight"] = torch.ones(1)
    path = directory / "bad.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError):
        load_vae(directory, "cpu", path)


def test_diffusers_directory_vae_loading(tiny_vae):
    model, directory = tiny_vae
    model.save_pretrained(directory)
    restored = load_vae(directory, "cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_comfy_decoder_width_override_is_explicit_and_strict(tiny_vae):
    import json

    model, directory = tiny_vae
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["decoder_block_out_channels"] = [4, 8]
    config_path.write_text(json.dumps(config))
    path = directory / "comfy.safetensors"
    save_file(_comfy_state(model), str(path))
    with pytest.warns(UserWarning, match="decoder widths.*override config"):
        restored = load_vae(directory, "cpu", path)
    assert list(restored.config.decoder_block_out_channels) == [8, 16]
    assert list(restored.config.block_out_channels) == [8, 16]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
