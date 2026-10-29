"""CPU CLI lifecycle tests without loading released models or allocating CUDA."""

import pytest
import torch
from PIL import Image
from torch import nn

import training.convrot as convrot
import training.quantization as quantization
import training.sample as sample
from training.lora import LoRALinear, inject_lora, save_lora
from training.memory import place_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8)])
        self.head = nn.Linear(8, 8)


@pytest.mark.parametrize("quant_mode", ["none", "int8", "int8_convrot"])
@pytest.mark.parametrize("with_lora", [False, True])
def test_cli_stages_vae_and_dit_and_quantizes_loaded_adapter(monkeypatch, tmp_path, quant_mode, with_lora):
    events = []
    model = TinyModel()
    adapter_file = tmp_path / "adapter.safetensors"
    if with_lora:
        source = TinyModel()
        inject_lora(source, rank=2, alpha=3, targets="blocks")
        with torch.no_grad():
            source.layers[0].lora_B.fill_(0.125)
        save_lora(source, adapter_file)

    def assert_adapter_loaded_and_frozen():
        assert not any(value.requires_grad for value in model.parameters())
        if with_lora:
            adapter = model.layers[0]
            assert isinstance(adapter, LoRALinear)
            assert adapter.rank == 2
            assert adapter.alpha == 9
            assert adapter.scale == 4.5
            torch.testing.assert_close(adapter.lora_B, torch.full_like(adapter.lora_B, 0.125))

    real_int8 = quantization.quantize_frozen_linears
    real_convrot = convrot.quantize_frozen_convrot_linears

    def int8_spy(value):
        assert value is model
        assert_adapter_loaded_and_frozen()
        events.append("quant:int8")
        return real_int8(value)

    def convrot_spy(value, group_size):
        assert value is model
        assert group_size == 4
        assert_adapter_loaded_and_frozen()
        events.append("quant:int8_convrot")
        return real_convrot(value, group_size)

    monkeypatch.setattr(quantization, "quantize_frozen_linears", int8_spy)
    monkeypatch.setattr(convrot, "quantize_frozen_convrot_linears", convrot_spy)
    monkeypatch.setattr(sample, "bundle_paths", lambda path: (path / "tokenizer", path / "text_encoder", path / "vae"))
    monkeypatch.setattr(sample, "encode_prompts", lambda *a, **kw: [torch.zeros(2, 8), torch.zeros(1, 8)])

    def load_model(path, dtype):
        assert dtype == torch.float32
        events.append("load_dit_cpu")
        assert all(value.device.type == "cpu" for value in model.parameters())
        return model

    monkeypatch.setattr(sample, "load_transformer", load_model)

    class FakeVAE:
        dtype = torch.float32

        def eval(self):
            return self

        def to(self, device):
            assert torch.device(device).type == "cpu"
            events.append("vae_to_cpu")
            return self

    vae = FakeVAE()
    monkeypatch.setattr(sample.AutoencoderKLFlux2, "from_pretrained", lambda *a, **kw: vae)

    def encode(value, pixels):
        assert value is vae
        assert "place:1" not in events
        events.append("encode_reference")
        return torch.ones(1, 4, 2, 2)

    monkeypatch.setattr(sample, "encode_pixels", encode)

    def placement(value, device, count):
        assert value is model
        assert torch.device(device).type == "cpu"
        if count == 1:
            assert events[-1] == "vae_to_cpu"
        else:
            assert events[-1] == "generate"
        events.append(f"place:{count}")
        place_model(value, device, count)

    monkeypatch.setattr(sample, "place_model", placement)

    def generate(value, positive, negative, references, *args):
        assert value is model
        assert events[-1] == "place:1"
        assert_adapter_loaded_and_frozen()
        base = value.layers[0].base if with_lora else value.layers[0]
        if quant_mode == "none":
            assert type(base) is nn.Linear
        elif quant_mode == "int8":
            assert type(base) is quantization.FrozenInt8Linear
        else:
            assert type(base) is convrot.FrozenConvRotLinear
            assert base.group_size == 4
        torch.testing.assert_close(references[0], torch.ones(1, 4, 2, 2))
        events.append("generate")
        return torch.zeros(1, 4, 2, 2)

    monkeypatch.setattr(sample, "generate", generate)

    def decode(value, latents):
        assert value is vae
        assert events[-2:] == ["place:0", "vae_to_cpu"]
        events.append("decode")
        return torch.zeros(1, 3, 32, 32)

    monkeypatch.setattr(sample, "decode_latents", decode)
    monkeypatch.setattr(sample, "save_rgb", lambda image, path: events.append("save"))
    reference_file = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference_file)
    argv = ["--bundle", str(tmp_path), "--prompt", "edit", "--out", str(tmp_path / "out.png"),
            "--device", "cpu", "--height", "32", "--width", "32", "--ref", str(reference_file),
            "--blocks-to-swap", "1", "--base-quantization", quant_mode, "--convrot-group-size", "4"]
    if with_lora:
        argv += ["--lora", str(adapter_file), "--alpha", "9"]
    sample.main(argv)
    quant_events = [event for event in events if event.startswith("quant:")]
    assert quant_events == ([] if quant_mode == "none" else [f"quant:{quant_mode}"])
    assert events.index("encode_reference") < events.index("place:1") < events.index("generate")
    assert events[-3:] == ["vae_to_cpu", "decode", "save"]
