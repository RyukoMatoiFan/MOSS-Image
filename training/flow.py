"""Flow-matching noise distribution and velocity target from the MOSS-Image report.

The final pretraining interval draws sigma on the ``T = 1000`` grid from a logit-normal
truncated at 0.95, rescaled onto ``(0, 1]``, mixed with 1% uniform mass. The velocity
target is ``epsilon - z_0`` at ``z_sigma = (1 - sigma) z_0 + sigma epsilon``. The DiT
sees timestep ``total * sigma``, which is ``1000 * sigma`` on the report's grid.
"""

from __future__ import annotations

import math

import torch


def flow_cdf(sigma: torch.Tensor, mu: float, width: float) -> torch.Tensor:
    """CDF of ``S = 1 - sigmoid(Z)`` for ``Z ~ Normal(mu, width^2)``.

    This is ``F`` in the technical report. ``F(0) = 0`` and ``F(x) = 1`` for ``x >= 1``.

    Args:
        sigma: Candidate values in real numbers. Values outside ``(0, 1)`` saturate.
        mu: Mean of the underlying normal.
        width: Standard deviation of the underlying normal. Must be positive.

    Returns:
        CDF values, same shape as ``sigma``, float64.
    """

    if width <= 0:
        raise ValueError("width must be positive.")
    values = torch.as_tensor(sigma, dtype=torch.float64)
    result = torch.zeros_like(values)
    valid = (values > 0) & (values < 1)
    logit = torch.log((1 - values[valid]) / values[valid])
    result[valid] = 1 - 0.5 * (1 + torch.erf((logit - mu) / (width * math.sqrt(2))))
    result[values >= 1] = 1
    return result


def discrete_sigma_probabilities(
    total: int = 1000,
    mu: float = -1.09,
    width: float = 0.95,
    cutoff: float = 0.95,
    uniform_mixture: float = 0.01,
) -> torch.Tensor:
    """Probability of each training sigma ``j / total`` for ``j = 1..total``.

    Args:
        total: Grid size ``T``. Sigma ``j / T`` is index ``j - 1`` in the result.
        mu: Logit-normal mean from the final pretraining interval.
        width: Logit-normal standard deviation.
        cutoff: Truncation applied before rescaling onto the full grid.
        uniform_mixture: Weight of the discrete uniform component.

    Returns:
        ``[total]`` probabilities that sum to 1.
    """

    if total < 1 or not 0 < cutoff <= 1 or not 0 <= uniform_mixture < 1:
        raise ValueError("Invalid flow-matching grid settings.")
    index = torch.arange(1, total + 1, dtype=torch.float64)
    upper = cutoff * index / total
    lower = cutoff * (index - 1) / total
    truncated = (flow_cdf(upper, mu, width) - flow_cdf(lower, mu, width)) / flow_cdf(
        torch.tensor(cutoff, dtype=torch.float64), mu, width
    )
    probabilities = (1 - uniform_mixture) * truncated + uniform_mixture / total
    return probabilities.to(dtype=torch.float64)


class FlowMatcher:
    """Samples the report's discrete sigma and builds the flow-matching regression target."""

    def __init__(
        self,
        total: int = 1000,
        mu: float = -1.09,
        width: float = 0.95,
        cutoff: float = 0.95,
        uniform_mixture: float = 0.01,
    ) -> None:
        self.total = total
        self.probabilities = discrete_sigma_probabilities(total, mu, width, cutoff, uniform_mixture)
        self.sigmas = torch.arange(1, total + 1, dtype=torch.float32) / total

    def sample_sigma(self, batch: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw one sigma per batch row.

        Args:
            batch: Number of independent sigmas.
            device: Destination device.
            generator: Optional torch generator. ``None`` uses the global RNG.

        Returns:
            ``[batch]`` float32 sigmas in ``{1/T, ..., 1}``.
        """

        sampling_device = generator.device if generator is not None else torch.device("cpu")
        choices = torch.multinomial(self.probabilities.float().to(sampling_device), batch,
                                    replacement=True, generator=generator)
        return self.sigmas.to(sampling_device)[choices].to(device)

    def noisy_latent(self, clean: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate toward noise and return the velocity target.

        Args:
            clean: ``z_0``, any shape starting with the batch.
            sigma: ``[batch]`` sigmas.
            noise: ``epsilon``, same shape as ``clean``.

        Returns:
            ``(z_sigma, epsilon - z_0)``. ``z_sigma`` uses the scheduler convention
            ``sigma * epsilon + (1 - sigma) * z_0``.
        """

        if noise.shape != clean.shape:
            raise ValueError("noise and clean latents must have the same shape.")
        view = sigma.to(clean.dtype).view(-1, *([1] * (clean.ndim - 1)))
        latent = view * noise + (1 - view) * clean
        return latent, noise - clean

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to the DiT timestep.

        Args:
            sigma: Sigmas in ``(0, 1]``.

        Returns:
            ``total * sigma`` on the same device, float32. With the default
            ``total`` of 1000 this is the report's ``1000 * sigma``.
        """

        return sigma.float() * self.total
