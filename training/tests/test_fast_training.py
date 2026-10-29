"""CPU checks for actual-MOSS fused updates across sequential singleton graphs."""

import copy

import torch

from training.fused_adafactor import FusedAdafactor
from training.model import MOSSImageTransformer


def test_real_moss_fused_singletons_match_deferred_full_step() -> None:
    # Seed the CPU generator without initializing CUDA.
    torch.random.default_generator.manual_seed(91)
    fused_model = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=2, text_dim=16,
        freq_dim=8, latent_channels=4, gradient_checkpointing=True,
    )
    reference_model = copy.deepcopy(fused_model)
    fused = FusedAdafactor(fused_model.named_parameters(), lr=1e-3).install_hooks()
    reference = FusedAdafactor(reference_model.named_parameters(), lr=1e-3)
    examples = [
        (torch.randn(1, 4, 2, 2), torch.randn(1, 3, 16),
         [torch.randn(1, 4, 1, 1)], torch.randn(1, 4, 2, 2)),
        (torch.randn(1, 4, 1, 3), torch.randn(1, 2, 16),
         [], torch.randn(1, 4, 1, 3)),
    ]
    total_count = sum(target.numel() for _, _, _, target in examples)
    try:
        for step in range(2):
            before = {name: parameter.detach().clone()
                      for name, parameter in fused_model.named_parameters()}
            fused.begin_step()
            reference.begin_step()
            for index, (latent, context, refs, target) in enumerate(examples):
                fused.set_step_enabled(index == len(examples) - 1)
                for model in (fused_model, reference_model):
                    prediction = model(
                        latent, torch.tensor([300.0]), context,
                        ref_hidden_states=refs,
                    )
                    ((prediction - target).square().sum() / total_count).backward()
                if index == 0:
                    assert all(torch.equal(parameter, before[name])
                               for name, parameter in fused_model.named_parameters())
            fused.finish_backward()
            reference.step()
            assert fused.global_step == reference.global_step == step + 1
            assert fused.updated_parameter_count == len(list(fused_model.parameters()))
            assert all(parameter.grad is None for parameter in fused_model.parameters())
            for actual, expected in zip(fused_model.parameters(), reference_model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(
                torch.tensor(fused.gradient_norm), torch.tensor(reference.gradient_norm),
                rtol=1e-5, atol=1e-7,
            )
            reference.zero_grad(set_to_none=True)
    finally:
        fused.close()
