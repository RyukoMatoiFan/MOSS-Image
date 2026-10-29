"""Single-file encoder loading must materialize weights and rotary buffers."""

import pytest
import torch
from safetensors.torch import save_file

from training.text import load_text_encoder


def test_dense_single_file_encoder_matches_original(tmp_path):
    from transformers import BertConfig, BertModel
    config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32)
    config.save_pretrained(tmp_path)
    original = BertModel(config).eval()
    weights = tmp_path / "encoder.safetensors"
    save_file(original.state_dict(), str(weights))
    loaded = load_text_encoder(tmp_path, torch.device("cpu"), torch.float32, weights)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).last_hidden_state, original(tokens).last_hidden_state)
    assert all(value.device.type == "cpu" for value in loaded.state_dict().values())


def test_comfy_qwen_single_file_restores_nonpersistent_rotary_buffers(tmp_path):
    qwen = pytest.importorskip("transformers.models.qwen3_5")
    config = qwen.Qwen3_5Config(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "layer_types": ["full_attention"],
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]}},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "depth": 1,
                       "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 16,
                       "patch_size": 2, "temporal_patch_size": 1, "spatial_merge_size": 1},
    )
    original = qwen.Qwen3_5Model(config).eval()
    config.save_pretrained(tmp_path)
    state = {"model." + key: value for key, value in original.state_dict().items()}
    state["lm_head.weight"] = torch.randn(64, 32)
    weights = tmp_path / "comfy-qwen.safetensors"
    save_file(state, str(weights))
    loaded = load_text_encoder(tmp_path, torch.device("cpu"), torch.float32, weights)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).last_hidden_state, original(tokens).last_hidden_state)
    assert all(value.device.type != "meta" for value in loaded.buffers())
