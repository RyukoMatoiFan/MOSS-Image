"""Offloaded gradients and updates must match resident training."""

import pytest
import torch

from training.memory import place_model
from training.model import MOSSImageTransformer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device copies"),
]


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_offloaded_edit_gradients_and_optimizer_update(precision):
    torch.manual_seed(19)
    kwargs = dict(dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                  text_dim=16, freq_dim=8, latent_channels=4, gradient_checkpointing=True)
    resident = MOSSImageTransformer(**kwargs).train()
    swapped = MOSSImageTransformer(**kwargs).train()
    swapped.load_state_dict(resident.state_dict())
    device = torch.device("cuda")
    place_model(resident, device, 0)
    place_model(swapped, device, 2)
    target = torch.randn(1, 4, 4, 4, device=device)
    context = torch.randn(1, 3, 16, device=device)
    refs = [torch.randn(1, 4, 2, 4, device=device)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3) for model in (resident, swapped)]
    for model, optimizer in zip((resident, swapped), optimizers):
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model(target, torch.tensor([500.], device=device), context, ref_hidden_states=refs)
                loss = output.float().square().mean()
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    resident.eval()
    swapped.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        first_output = resident(target, torch.tensor([500.], device=device), context, ref_hidden_states=refs)
        second_output = swapped(target, torch.tensor([500.], device=device), context, ref_hidden_states=refs)
    torch.testing.assert_close(first_output, second_output, atol=2e-5, rtol=2e-3)
    for first, second in zip(resident.parameters(), swapped.parameters()):
        torch.testing.assert_close(first.grad.cpu(), second.grad.cpu(), atol=2e-5, rtol=2e-3)
        torch.testing.assert_close(first.cpu(), second.cpu(), atol=2e-5, rtol=2e-3)
    assert all(p.device.type == "cpu" for p in swapped.layers.parameters())
