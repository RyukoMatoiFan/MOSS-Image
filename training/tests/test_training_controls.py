"""CPU regressions for training controls, cache identity, and checkpoint output."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

import training.precache as precache_module
import training.train as trainer
import training.validation as validation_module
from training.cache import cache_fingerprint, cache_provenance, write_example, write_provenance
from training.config import TrainConfig, load_config
from training.optimizers import parameter_groups
from training.safetensors_stream import save_stream
from training.token_budget import index_tokens, prepared_tokens


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.num_layers = 0


def test_exported_training_config_remains_loadable(tmp_path):
    config = TrainConfig(model="relative-base", packed_batching=True, checkpoint_fraction=0.5, lr=1e-5)
    trainer._record_config(config, tmp_path)
    loaded = load_config(tmp_path / "train_config.json")
    assert loaded.model == str(Path("relative-base").resolve())
    assert loaded.packed_batching and loaded.checkpoint_fraction == 0.5
    assert loaded.lr == 1e-5 and isinstance(loaded.lr, float)


def _install_tiny_trainer(monkeypatch, *, eval_rows: bool = False):
    """Keep the real control loop while replacing model and data costs."""
    model = _TinyTrainModel()
    loaded_paths = []
    saved_steps = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def load(path, **kwargs):
        loaded_paths.append(Path(path))
        return model

    monkeypatch.setattr(trainer, "load_transformer", load)
    monkeypatch.setattr(trainer, "place_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "configure_model_performance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer,
        "_load_rows",
        lambda config: ([{"split": "train"}], ([{"split": "eval"}] if eval_rows else []), torch.zeros(1, 1)),
    )
    monkeypatch.setattr(
        trainer,
        "_prepare_batch",
        lambda rows, empty, config, train: [(torch.ones(1, 1, 1), torch.ones(1, 1), [])],
    )

    def squared_error(current, matcher, batch, device):
        return current.weight.square(), sum(example[0].numel() for example in batch)

    monkeypatch.setattr(trainer, "_group_squared_error", squared_error)
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda model, optimizer, step, config, directory: saved_steps.append(step),
    )
    return model, loaded_paths, saved_steps


def test_resume_requires_training_state_but_warm_start_runs_weights_only(monkeypatch, tmp_path):
    resume = tmp_path / "resume"
    resume.mkdir()
    loaded = []
    monkeypatch.setattr(trainer, "load_transformer", lambda path, **kwargs: loaded.append(path))
    with pytest.raises(FileNotFoundError, match="resume requires train_state.pt.*warm_start"):
        trainer.train(TrainConfig(model="", mode="full", resume=str(resume), steps=1))
    assert loaded == []

    _, loaded_paths, saved_steps = _install_tiny_trainer(monkeypatch)
    warm_start = tmp_path / "weights-only"
    config = TrainConfig(
        model="",
        mode="full",
        warm_start=str(warm_start),
        output_dir=str(tmp_path / "run"),
        steps=1,
        ckpt_every=10,
        log_every=10,
    )
    trainer.train(config)
    assert loaded_paths == [warm_start]
    assert saved_steps == [0]


@pytest.mark.parametrize("change,message", [
    ({"model": "other-base"}, "frozen base"),
    ({"base_quantization": "none"}, "base_quantization"),
    ({"convrot_group_size": 32}, "convrot_group_size"),
    ({"adam_betas": (0.8, 0.9)}, "adam_betas"),
])
def test_lora_resume_rejects_changed_base_and_optimizer_before_loading(tmp_path, monkeypatch, change, message):
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    (resume / "train_state.pt").write_bytes(b"not reached")
    original = TrainConfig(model="original-base", mode="lora", base_quantization="int8_convrot")
    (resume / "train_config.json").write_text(json.dumps(asdict(original)))
    config = TrainConfig(**(asdict(original) | change | {"resume": str(resume)}))
    monkeypatch.setattr(trainer, "load_transformer", lambda *a, **k: pytest.fail("model load must not be reached"))
    with pytest.raises(ValueError, match=message):
        trainer.train(config)


def test_checkpoint_failure_cleans_staging_and_never_publishes_final_path(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "save_lora", lambda model, path: path.write_bytes(b"adapter"))

    def fail_save(*args, **kwargs):
        raise OSError("simulated state write failure")

    monkeypatch.setattr(trainer.torch, "save", fail_save)
    model = _TinyTrainModel()
    optimizer = torch.optim.AdamW(model.parameters())
    destination = tmp_path / "run" / "step_000001"
    config = TrainConfig(mode="lora", cache_dir=str(tmp_path / "cache"))
    with pytest.raises(OSError, match="simulated state write failure"):
        trainer.save_checkpoint(model, optimizer, 0, config, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".step_000001.*.partial"))


def test_evaluation_uses_completed_step_cadence_independent_of_logging(monkeypatch, tmp_path):
    _, _, saved_steps = _install_tiny_trainer(monkeypatch, eval_rows=True)
    evaluations = []

    def evaluate(*args, **kwargs):
        evaluations.append(True)
        return {"eval_loss": float(len(evaluations))}

    monkeypatch.setattr(validation_module, "evaluate_fixed", evaluate)
    config = TrainConfig(
        model="tiny",
        mode="full",
        output_dir=str(tmp_path / "run"),
        steps=4,
        eval_every=2,
        log_every=3,
        ckpt_every=99,
    )
    trainer.train(config)
    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(evaluations) == 2
    assert [record["step"] for record in records] == [0, 1, 3]
    assert [record["step"] for record in records if "eval_loss" in record] == [1, 3]
    assert saved_steps == [3]


def test_fixed_validation_is_repeatable_and_preserves_rng_and_model_mode():
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, value):
            return value * self.scale

    model = Scale().train()
    prepared = [(torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), torch.zeros(2, 1), [])]

    def score(current, matcher, examples, device):
        clean = examples[0][0].unsqueeze(0)
        sigma = matcher.sample_sigma(1, device)
        ordinary_noise = torch.randn_like(clean)
        noisy, target = matcher.noisy_latent(clean, sigma, ordinary_noise)
        residual = current(noisy) - target
        return residual.square().sum(), residual.numel()

    torch.manual_seed(881)
    before = torch.get_rng_state().clone()
    first = validation_module.evaluate_fixed(
        model, prepared, (0.25, 0.75), 17, 1000, torch.device("cpu"), "no", score
    )
    middle = torch.get_rng_state().clone()
    second = validation_module.evaluate_fixed(
        model, prepared, (0.25, 0.75), 17, 1000, torch.device("cpu"), "no", score
    )
    after = torch.get_rng_state()
    assert first == second
    assert torch.equal(before, middle) and torch.equal(before, after)
    assert model.training


def test_precache_uses_exif_oriented_pixel_dimensions(monkeypatch, tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Rotate a stored 48x32 image to displayed 32x48.
    Image.new("RGB", (48, 32), color=(20, 40, 60)).save(path, exif=exif)
    seen = []

    def encode(vae, pixels):
        seen.append(tuple(pixels.shape))
        return torch.zeros(1, 4, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    monkeypatch.setattr(precache_module, "encode_pixels", encode)
    latent, height, width = precache_module._encode_image(
        object(), path, 64 * 64, torch.device("cpu")
    )
    assert (height, width) == (64, 48)
    assert seen == [(1, 3, 64, 48)]
    assert latent.shape == (4, 4, 3)


def _provenance_fixture(root: Path):
    tokenizer = root / "tokenizer"
    text = root / "text"
    vae = root / "vae"
    for directory in (tokenizer, text, vae):
        directory.mkdir()
        (directory / "config.json").write_text(f'{{"name":"{directory.name}"}}', encoding="utf-8")
    target = root / "target.png"
    reference = root / "reference.png"
    target.write_bytes(b"target-a")
    reference.write_bytes(b"reference-a")
    jobs = [{"target": target, "refs": [reference], "prompt": "edit"}]
    return tokenizer, text, vae, target, jobs


def test_cache_fingerprint_tracks_source_and_config_content(tmp_path):
    tokenizer, text, vae, target, jobs = _provenance_fixture(tmp_path)
    template = tokenizer / "chat_template.jinja"
    template.write_text("original template")
    initial = cache_provenance(tokenizer, text, vae, jobs, 64, text_config=text, vae_config=vae)
    template.write_text("changed template")
    changed_template = cache_provenance(tokenizer, text, vae, jobs, 64, text_config=text, vae_config=vae)
    assert changed_template["fingerprint"] != initial["fingerprint"]
    template.write_text("original template")
    changed_dtype = cache_provenance(tokenizer, text, vae, jobs, 64, text_config=text, vae_config=vae,
                                     text_dtype="bfloat16")
    assert changed_dtype["fingerprint"] != initial["fingerprint"]
    target.write_bytes(b"target-b")
    source_changed = cache_provenance(tokenizer, text, vae, jobs, 64, text_config=text, vae_config=vae)
    assert source_changed["fingerprint"] != initial["fingerprint"]
    target.write_bytes(b"target-a")
    (text / "config.json").write_text('{"name":"changed"}', encoding="utf-8")
    config_changed = cache_provenance(tokenizer, text, vae, jobs, 64, text_config=text, vae_config=vae)
    assert config_changed["fingerprint"] != initial["fingerprint"]

    cache = tmp_path / "cache"
    cache.mkdir()
    write_provenance(cache, initial)
    assert cache_fingerprint(cache) == initial["fingerprint"]
    with pytest.raises(ValueError, match="fingerprint does not match"):
        cache_fingerprint(cache, source_changed["fingerprint"])


def test_resume_rejects_a_different_cache_fingerprint(monkeypatch, tmp_path):
    _install_tiny_trainer(monkeypatch)
    resume = tmp_path / "resume"
    resume.mkdir()
    torch.save(
        {
            "step": 0,
            "optimizer": {},
            "cache_fingerprint": "old-cache",
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        },
        resume / "train_state.pt",
    )
    monkeypatch.setattr(trainer, "cache_fingerprint", lambda *args, **kwargs: "current-cache")
    config = TrainConfig(
        model="",
        mode="full",
        resume=str(resume),
        output_dir=str(tmp_path / "run"),
        steps=2,
    )
    with pytest.raises(ValueError, match="Resume cache fingerprint differs"):
        trainer.train(config)


def test_token_budget_counts_target_references_and_real_or_empty_text(tmp_path):
    cache = tmp_path / "cache"
    path = cache / "example.safetensors"
    target = torch.zeros(4, 2, 3)
    context = torch.zeros(5, 8)
    references = [torch.zeros(4, 1, 4), torch.zeros(4, 2, 2)]
    write_example(path, target, context, references, "edit", 32, 48)
    example = (target, context, references)
    assert prepared_tokens(example) == 6 + 5 + 4 + 4
    assert index_tokens({"file": path.name}, cache, empty_length=7) == 6 + 7 + 4 + 4


def test_adam_betas_and_role_based_decay_groups_are_preserved():
    named = [
        ("layers.0.proj.q.weight", nn.Parameter(torch.ones(2, 2))),
        ("layers.0.proj.q.bias", nn.Parameter(torch.ones(2))),
        ("video_embedder.weight", nn.Parameter(torch.ones(2, 2))),
        ("layers.0.other_norm.weight", nn.Parameter(torch.ones(2))),
        ("layers.1.q_norm.weight", nn.Parameter(torch.ones(2))),
    ]
    groups = parameter_groups(
        named,
        0.2,
        "matrix",
        qk_norm_decay_to_one=True,
        depth=2,
    )
    by_name = {name: group for group in groups for name in group["names"]}
    assert by_name["layers.0.proj.q.weight"]["weight_decay"] == 0.2
    for name in ("layers.0.proj.q.bias", "video_embedder.weight", "layers.0.other_norm.weight"):
        assert by_name[name]["weight_decay"] == 0.0
        assert not by_name[name]["decay_to_one"]
    final_norm = by_name["layers.1.q_norm.weight"]
    assert final_norm["weight_decay"] == 0.0
    assert final_norm["decay_to_one"] and final_norm["decay_to_one_coef"] == 0.2

    config = TrainConfig(optimizer="adamw", weight_decay=0.2, adam_betas=(0.8, 0.95))
    optimizer = trainer._build_optimizer(groups, config)
    assert all(group["betas"] == (0.8, 0.95) for group in optimizer.param_groups)


def test_streamed_safetensors_are_exactly_loadable_by_standard_reader(tmp_path):
    path = tmp_path / "streamed.safetensors"
    first = torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16)
    second = torch.tensor([[3.0, 4.5], [-7.0, 8.0]], dtype=torch.bfloat16)
    labels = torch.tensor([5, -2, 9], dtype=torch.int64)
    flags = torch.tensor([True, False, True], dtype=torch.bool)
    save_stream(
        {
            "joined": ([3, 2], [first, second]),
            "labels": ([3], [labels]),
            "flags": ([3], [flags]),
        },
        path,
        metadata={"format": "pt", "layout": "test"},
    )
    loaded = load_file(str(path))
    assert torch.equal(loaded["joined"], torch.cat([first, second]))
    assert torch.equal(loaded["labels"], labels)
    assert torch.equal(loaded["flags"], flags)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        assert reader.metadata() == {"format": "pt", "layout": "test"}
