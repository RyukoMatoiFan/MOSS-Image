"""Exercise the real cache, trainer, checkpoint and resume path on CUDA."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from training.cache import write_example, write_index
from training.config import TrainConfig
from training.model import MOSSImageTransformer
from training.train import train
from training.weights import save_transformer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("mode,quantization,swap_mode", [
    ("full", "none", "sync"), ("lora", "int8", "sync"), ("lora", "int8_convrot", "sync"),
    ("lora", "none", "async"), ("lora", "int8", "async"), ("lora", "int8_convrot", "async"),
])
def test_offloaded_training_save_and_resume(tmp_path, mode, quantization, swap_mode):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                                text_dim=16, freq_dim=8, latent_channels=4)
    save_transformer(model, bundle / "diffusion_pytorch_model.safetensors")
    cache = tmp_path / "cache"
    write_example(cache / "0.safetensors", torch.randn(4, 4, 4), torch.randn(3, 16),
                  [torch.randn(4, 2, 4)], "edit", 64, 64)
    write_index(cache, [{"file": "0.safetensors", "latent": [4, 4, 4],
                        "refs": [[4, 2, 4]], "height": 64, "width": 64}])
    save_file({"context": torch.randn(2, 16)}, str(cache / "empty.safetensors"))
    config = TrainConfig(model=str(bundle), cache_dir=str(cache), output_dir=str(tmp_path / "run"),
                         mode=mode, base_quantization=quantization, blocks_to_swap=2,
                         block_swap_mode=swap_mode,
                         mixed_precision="bf16", steps=3, accum=2, ckpt_every=2, log_every=1,
                         resolution=64, rank=2)
    train(config)
    checkpoint = tmp_path / "run" / "step_000002"
    assert (checkpoint / "train_state.pt").is_file()
    final = tmp_path / "run" / "step_000003"
    filename = "lora.safetensors" if mode == "lora" else "diffusion_pytorch_model.safetensors"
    uninterrupted = load_file(str(final / filename))
    config.resume = str(checkpoint)
    config.output_dir = str(tmp_path / "resumed")
    train(config)
    resumed = load_file(str(tmp_path / "resumed" / "step_000003" / filename))
    for name in uninterrupted:
        torch.testing.assert_close(uninterrupted[name], resumed[name], rtol=1e-5, atol=1e-6)
