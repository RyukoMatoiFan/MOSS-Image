"""Production trainer integration of packed, offloaded and quantized paths."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from training.cache import write_example, write_index
from training.config import TrainConfig
from training.model import MOSSImageTransformer
from training.train import train
from training.weights import save_transformer

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires reserved CUDA")]


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_packed_trainer_with_memory_controls_saves_and_resumes(tmp_path, mode):
    torch.manual_seed(267)
    bundle = tmp_path / "model"
    bundle.mkdir()
    model = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=3,
                                 text_dim=16, freq_dim=8, latent_channels=4)
    save_transformer(model, bundle / "diffusion_pytorch_model.safetensors")
    cache = tmp_path / "cache"
    rows = []
    for index, shape in enumerate(((4, 2, 3), (4, 3, 2), (4, 2, 2))):
        refs = [torch.randn(4, 1, index + 1)] if index != 1 else []
        write_example(cache / f"{index}.safetensors", torch.randn(shape), torch.randn(2 + index, 16),
                      refs, "tiny", shape[1] * 16, shape[2] * 16)
        rows.append({"file": f"{index}.safetensors", "latent": list(shape),
                     "height": shape[1] * 16, "width": shape[2] * 16,
                     "refs": [list(ref.shape) for ref in refs]})
    write_index(cache, rows)
    save_file({"context": torch.randn(2, 16)}, str(cache / "empty.safetensors"))
    config = TrainConfig(model=str(bundle), cache_dir=str(cache), output_dir=str(tmp_path / "run"),
                         mode=mode, resolution=64, rank=2, alpha=2, targets="joint", batch=2, accum=2,
                         steps=2, warmup=0, log_every=3, ckpt_every=1, n_eval_holdout=1,
                         eval_every=1, eval_sigmas=(0.25, 0.75), packed_batching=True, max_tokens=32,
                         checkpoint_fraction=0.5, activation_offload=True, mixed_precision="bf16",
                         attention_backend="flash_attn_2", blocks_to_swap=2,
                         block_swap_mode="async" if mode == "lora" else "async_full",
                         base_quantization="int8_convrot" if mode == "lora" else "none",
                         convrot_group_size=16, quantized_backend="triton" if mode == "lora" else "eager",
                         optimizer="adamw" if mode == "lora" else "adafactor_fused",
                         full_parameter_dtype="fp32" if mode == "lora" else "bf16",
                         gradient_accumulation_device="device" if mode == "lora" else "cpu",
                         weight_decay_policy="matrix", qk_norm_decay_to_one=True,
                         adam_betas=(0.9, 0.95), grad_clip=1.0 if mode == "lora" else 0)
    train(config)
    filename = "lora.safetensors" if mode == "lora" else "diffusion_pytorch_model.safetensors"
    expected = load_file(str(tmp_path / "run" / "step_000002" / filename))
    config.resume = str(tmp_path / "run" / "step_000001")
    config.output_dir = str(tmp_path / "resumed")
    train(config)
    actual = load_file(str(tmp_path / "resumed" / "step_000002" / filename))
    for name, tensor in expected.items():
        assert torch.isfinite(actual[name]).all()
        torch.testing.assert_close(actual[name], tensor, rtol=0.01 if mode == "full" else 1e-4,
                                   atol=0.002 if mode == "full" else 1e-5)
    logs = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logs if "eval_loss" in row] == [0, 1]
    state = torch.load(tmp_path / "resumed" / "step_000002" / "train_state.pt", weights_only=False)
    assert state["step"] == 1
    assert not list((tmp_path / "resumed").glob("*.partial"))
