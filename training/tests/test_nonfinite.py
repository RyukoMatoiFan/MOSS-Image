"""A bad accumulated micro-batch must never reach an optimizer update or save."""

import json

import pytest
import torch
from safetensors.torch import save_file

import training.train as trainer
from training.cache import write_example, write_index
from training.config import TrainConfig
from training.model import MOSSImageTransformer


class InjectNonfinite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, kind, bad_value, backward_calls):
        ctx.kind = kind
        ctx.bad_value = bad_value
        ctx.backward_calls = backward_calls
        ctx.shape = parameter.shape
        return parameter.new_tensor(bad_value if kind == "loss" else 1.0)

    @staticmethod
    def backward(ctx, gradient):
        ctx.backward_calls.append(ctx.kind)
        return gradient.new_full(ctx.shape, ctx.bad_value), None, None, None


def setup_training(monkeypatch, tmp_path, grad_clip, kind, bad_value):
    # Force deterministic CPU device selection.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=1,
                                text_dim=16, freq_dim=8, latent_channels=4)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    monkeypatch.setattr(trainer, "load_transformer", lambda *a, **kw: model)
    cache = tmp_path / "cache"
    write_example(cache / "0.safetensors", torch.ones(4, 2, 2), torch.ones(2, 16),
                  [], "tiny", 32, 32)
    write_index(cache, [{"file": "0.safetensors", "latent": [4, 2, 2],
                         "refs": [], "height": 32, "width": 32}])
    save_file({"context": torch.zeros(1, 16)}, str(cache / "empty.safetensors"))
    config = TrainConfig(model="mock-loaded-model", mode="full", cache_dir=str(cache),
                         output_dir=str(tmp_path / "run"), resolution=32, steps=1,
                         accum=2, grad_clip=grad_clip, ckpt_every=1, log_every=1)
    optimizer_steps, saved, backward_calls, microbatches = [], [], [], []

    class TrackingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            optimizer_steps.append(True)
            return super().step(closure)

    monkeypatch.setattr(trainer, "_build_optimizer",
                        lambda parameters, config: TrackingAdamW(parameters, lr=config.lr))
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args: saved.append(args))

    def error_function(model, matcher, batch, device):
        parameter = next(model.parameters())
        count = sum(example[0].numel() for example in batch)
        microbatches.append(True)
        if len(microbatches) == 1 or kind == "finite":
            return parameter.square().sum(), count
        # The preceding microbatch has already accumulated real, finite gradients.
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
        return InjectNonfinite.apply(parameter, kind, bad_value, backward_calls), count

    monkeypatch.setattr(trainer, "_group_squared_error", error_function)
    return config, model, initial, optimizer_steps, saved, backward_calls, microbatches


@pytest.mark.parametrize("grad_clip", [0.0, 1.0])
@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")], ids=["inf", "nan"])
@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_nonfinite_accumulation_cancels_step_and_checkpoint(monkeypatch, tmp_path, grad_clip, bad_value, kind):
    config, model, initial, optimizer_steps, saved, backward_calls, microbatches = setup_training(
        monkeypatch, tmp_path, grad_clip, kind, bad_value,
    )
    error = FloatingPointError if kind == "loss" else RuntimeError
    message = "Nonfinite training loss" if kind == "loss" else "non-finite"
    with pytest.raises(error, match=message):
        trainer.train(config)
    assert len(microbatches) == 2
    assert not optimizer_steps
    assert not saved
    assert backward_calls == ([] if kind == "loss" else ["gradient"])
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    output = tmp_path / "run"
    assert not list(output.glob("step_*"))
    assert not (output / "metrics.jsonl").exists()


@pytest.mark.parametrize("grad_clip", [0.0, 1.0])
def test_finite_accumulation_updates_and_logs_gradient_norm(monkeypatch, tmp_path, grad_clip):
    config, model, initial, optimizer_steps, saved, _, microbatches = setup_training(
        monkeypatch, tmp_path, grad_clip, "finite", 0.0,
    )
    trainer.train(config)
    assert len(microbatches) == 2
    assert optimizer_steps == [True]
    assert len(saved) == 1
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters())
    record = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert record["grad_norm"] > 0
    assert torch.isfinite(torch.tensor(record["grad_norm"]))
