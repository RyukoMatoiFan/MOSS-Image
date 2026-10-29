"""Bounded CUDA peak-allocation regression for full-training block offload."""

import gc

import pytest
import torch

from training.memory import place_model
from training.model import MOSSImageTransformer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA memory measurements"),
]


def _full_training_peak(blocks_to_swap):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(43)
    model = MOSSImageTransformer(
        dim=128, ffn_dim=256, num_heads=4, num_layers=8,
        text_dim=64, freq_dim=32, latent_channels=16,
        gradient_checkpointing=True,
    ).train()
    place_model(model, torch.device("cuda"), blocks_to_swap)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    target = torch.randn(1, 16, 4, 4, device="cuda")
    context = torch.randn(1, 4, 64, device="cuda")
    timestep = torch.tensor([500.], device="cuda")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(target, timestep, context)
        loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        del output, loss
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    # State creation must actually have happened, including CPU block masters.
    assert len(optimizer.state) == len(list(model.parameters()))
    del optimizer, model, target, context, timestep
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return peak


def test_full_training_block_offload_reduces_cuda_peak(record_property):
    """Include backward and two AdamW steps; allocated bytes exclude idle baseline."""
    resident = _full_training_peak(blocks_to_swap=0)
    swapped = _full_training_peak(blocks_to_swap=8)
    record_property("resident_peak_bytes", resident)
    record_property("offloaded_peak_bytes", swapped)
    print(f"CUDA full-training peak allocated: resident={resident / 2**20:.2f} MiB, "
          f"offloaded={swapped / 2**20:.2f} MiB, ratio={swapped / resident:.3f}")
    assert swapped < resident * 0.8
