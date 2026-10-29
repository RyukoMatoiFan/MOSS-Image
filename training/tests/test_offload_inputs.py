"""Compatibility coverage for packed block inputs across offload strategies."""

import copy

import pytest
import torch
from torch import nn

from training.memory import offloaded_block, place_model


class _FourInputBlock(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        boundary_term = cu_seqlens[-1].to(tokens.dtype) * 0.01
        return self.projection(tokens + rope) * modulation + boundary_term


def test_synchronous_offloaded_block_forwards_optional_fourth_tensor() -> None:
    torch.manual_seed(1201)
    reference = _FourInputBlock()
    offloaded = copy.deepcopy(reference)
    reference_tokens = torch.randn(1, 7, 6, requires_grad=True)
    offloaded_tokens = reference_tokens.detach().clone().requires_grad_(True)
    rope = torch.randn(1, 7, 6)
    modulation = torch.randn(1, 1, 6)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    expected = reference(reference_tokens, rope, modulation, cu_seqlens)
    actual = offloaded_block(offloaded, offloaded_tokens, rope, modulation, cu_seqlens)
    expected.square().mean().backward()
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(offloaded_tokens.grad, reference_tokens.grad, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(offloaded.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)


class _FrozenFourInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FourInputBlock()])
        self.head = nn.Linear(6, 2)
        self.blocks_to_swap = 0
        self.block_swap_manager = None

    def forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if self.block_swap_manager is None:
            tokens = self.layers[0](tokens, rope, modulation, cu_seqlens)
        else:
            tokens = self.block_swap_manager.block(
                0,
                tokens,
                rope,
                modulation,
                cu_seqlens,
                forward_prefetch=None,
                backward_prefetch=None,
                checkpoint_block=True,
            )
        return self.head(tokens)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a reserved CUDA device")
def test_frozen_async_manager_forwards_optional_fourth_tensor() -> None:
    torch.manual_seed(1237)
    reference = _FrozenFourInputModel()
    streamed = copy.deepcopy(reference)
    for parameter in reference.layers.parameters():
        parameter.requires_grad_(False)
    for parameter in streamed.layers.parameters():
        parameter.requires_grad_(False)
    reference.cuda()
    place_model(streamed, torch.device("cuda"), 1, block_swap_mode="async")
    manager = streamed.block_swap_manager
    assert manager is not None
    reference_tokens = torch.randn(1, 7, 6, device="cuda", requires_grad=True)
    streamed_tokens = reference_tokens.detach().clone().requires_grad_(True)
    rope = torch.randn(1, 7, 6, device="cuda")
    modulation = torch.randn(1, 1, 6, device="cuda")
    cu_seqlens = torch.tensor([0, 2, 7], dtype=torch.int32, device="cuda")

    expected = reference(reference_tokens, rope, modulation, cu_seqlens)
    actual = streamed(streamed_tokens, rope, modulation, cu_seqlens)
    assert len(manager.owned_cuda_buffers()) == 1
    assert manager.owned_cuda_buffers()[0].device.type == "cuda"
    expected.square().mean().backward()
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(streamed_tokens.grad, reference_tokens.grad, rtol=1e-6, atol=1e-7)
    for actual_parameter, expected_parameter in zip(streamed.head.parameters(), reference.head.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-6, atol=1e-7)
    manager.close()
