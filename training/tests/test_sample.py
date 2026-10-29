"""Sampling must retain edit conditioning and work with offloaded blocks."""

import pytest
import torch

from training.memory import place_model
from training.model import MOSSImageTransformer
from training.sample import generate


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device copies")
@pytest.mark.parametrize("quantization", ["none", "int8", "int8_convrot"])
@pytest.mark.parametrize("swap_mode", ["sync", "async"])
def test_edit_sampling_matches_with_swapped_blocks(quantization, swap_mode):
    torch.manual_seed(23)
    model = MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                                text_dim=16, freq_dim=8, latent_channels=4).eval()
    model.requires_grad_(False)
    if quantization == "int8":
        from training.quantization import quantize_frozen_linears

        quantize_frozen_linears(model)
    elif quantization == "int8_convrot":
        from training.convrot import quantize_frozen_convrot_linears

        quantize_frozen_convrot_linears(model)
    device = torch.device("cuda")
    positive = torch.randn(3, 16)
    negative = torch.randn(2, 16)
    references = [torch.randn(1, 4, 2, 4)]
    kwargs = dict(model=model, prompt_states=positive, negative_states=negative,
                  references=references, height=64, width=64, steps=3, guidance=3.,
                  shift=5., skip_layers=(1,), seed=17, device=device)
    place_model(model, device, 0)
    resident = generate(**kwargs)
    place_model(model, device, 2, block_swap_mode=swap_mode)
    swapped = generate(**kwargs)
    torch.testing.assert_close(resident, swapped, rtol=0, atol=0)
    without_reference = generate(**{**kwargs, "references": None})
    assert not torch.equal(swapped, without_reference)


def test_cfg_keeps_references_in_both_branches():
    class Recorder(torch.nn.Module):
        latent_channels = 4

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.references = []

        def forward(self, latents, timestep, context, mask, **kwargs):
            self.references.append(kwargs["ref_hidden_states"])
            return torch.zeros_like(latents)

    model = Recorder()
    reference = torch.randn(1, 4, 2, 2)
    generate(model, torch.zeros(2, 16), torch.zeros(1, 16), [reference],
             32, 32, 2, 3., 5., (), 0, torch.device("cpu"))
    assert len(model.references) == 4
    for refs in model.references:
        torch.testing.assert_close(refs[0], reference)
