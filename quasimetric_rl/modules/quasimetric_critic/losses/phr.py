"""Shared Powell–Hestenes–Rockafellar inequality term."""
import torch


def phr_terms(residual: torch.Tensor, multiplier: torch.Tensor, rho: float):
    """Mean convention: effective per-edge sum coefficient is effective / E.

    Multipliers are held fixed in the primal step. This function does not
    prescribe how a functional or tabular dual is updated.
    """
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    multiplier = multiplier.detach()
    if rho == 0:
        return (multiplier * residual).mean(), multiplier
    effective = (multiplier + rho * residual).relu()
    return (effective.square() - multiplier.square()).mean() / (2 * rho), effective


class TabularPHR(torch.nn.Module):
    """Independent replay-row multipliers, learned only from edge residuals."""
    def __init__(self, edges: int, rho: float = 10., dual_step: float = .1):
        super().__init__()
        if edges <= 0 or rho <= 0 or dual_step <= 0:
            raise ValueError("edges, rho and dual_step must be positive")
        self.rho, self.dual_step = rho, dual_step
        self.register_buffer("multiplier", torch.zeros(edges))

    def forward(self, residual):
        if residual.shape != self.multiplier.shape:
            raise ValueError("PHR requires every replay edge in stable row order")
        return phr_terms(residual, self.multiplier, self.rho)[0]

    @torch.no_grad()
    def update(self, residual):
        if residual.shape != self.multiplier.shape or not torch.isfinite(residual).all():
            raise ValueError("invalid full replay residual")
        self.multiplier.add_(residual, alpha=self.dual_step).clamp_min_(0)
