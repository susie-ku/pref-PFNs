"""
GP preference model agent (Chu & Ghahramani 2005 style).

Posterior over f is approximated via Laplace approximation under a
Thurstone-Mosteller (probit) preference likelihood.

Model:
    f ~ GP(0, k_RBF(x, x'))
    P(x_i > x_j | f) = Phi((f_i - f_j) / sqrt(2))

The agent works only on a one-dimensional discrete candidate grid.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import Comparison, PBOAgent, Point, candidate_matrix


_SQRT2 = math.sqrt(2.0)


def _probit(z: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(z / _SQRT2))


def _log_probit(z: Tensor) -> Tensor:
    return torch.special.log_ndtr(z)


def _pref_log_lik(
    f: Tensor,
    winner_idx: list[int],
    loser_idx: list[int],
) -> Tensor:
    """Return the summed probit preference log-likelihood."""
    if not winner_idx:
        return torch.tensor(0.0, dtype=f.dtype, device=f.device)
    fw = f[winner_idx]
    fl = f[loser_idx]
    z = (fw - fl) / _SQRT2
    return _log_probit(z).sum()


def _rbf_kernel(x: Tensor, lengthscale: float, outputscale: float) -> Tensor:
    """Build an RBF covariance matrix for a one-dimensional grid."""
    x = x.unsqueeze(-1)
    sq_dist = (x - x.T) ** 2
    return outputscale * torch.exp(-0.5 * sq_dist / lengthscale**2)


def _one_dimensional_grid(candidate_pool: Tensor) -> Tensor:
    """Normalize scalar grids and reject unsupported multidimensional grids."""
    grid = candidate_matrix(candidate_pool)
    if grid.shape[-1] != 1:
        raise ValueError(
            "GPPBOAgent supports only one-dimensional grid candidate pools; "
            f"got shape {tuple(candidate_pool.shape)}."
        )
    return grid[:, 0]


def _scalar_point(point: Point, *, like: Tensor) -> Tensor:
    value = torch.as_tensor(point, dtype=like.dtype, device=like.device).reshape(-1)
    if value.numel() != 1:
        raise ValueError("GPPBOAgent comparisons must contain one-dimensional points.")
    return value[0]


def _laplace_posterior(
    x_pool: Tensor,
    winner_idx: list[int],
    loser_idx: list[int],
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    jitter: float = 1e-4,
    n_iter: int = 50,
    lr: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """Return the Laplace posterior mean and covariance on ``x_pool``."""
    n = x_pool.shape[0]
    K = _rbf_kernel(x_pool, lengthscale, outputscale)
    K = K + jitter * torch.eye(n, dtype=K.dtype, device=K.device)
    K_inv = torch.linalg.inv(K)

    if not winner_idx:
        return torch.zeros(n, dtype=K.dtype, device=K.device), K

    f = torch.zeros(n, dtype=K.dtype, device=K.device)
    sqrt_2pi = math.sqrt(2 * math.pi)
    for _ in range(n_iter):
        f = f.detach().requires_grad_(True)
        ll = _pref_log_lik(f, winner_idx, loser_idx)
        lp = -0.5 * (f @ K_inv @ f)
        objective = ll + lp
        objective.backward()
        with torch.no_grad():
            grad = f.grad.nan_to_num(0.0).clamp(-10.0, 10.0)
            f = f + lr * grad

    f_map = f.detach()

    W = torch.zeros(n, dtype=K.dtype, device=K.device)
    if winner_idx:
        fw = f_map[winner_idx]
        fl = f_map[loser_idx]
        z = (fw - fl) / _SQRT2
        phi = torch.exp(-0.5 * z**2) / sqrt_2pi
        Phi = _probit(z).clamp(1e-8, 1 - 1e-8)
        lam = ((phi / Phi) ** 2).nan_to_num(0.0)
        for idx, (winner, loser) in enumerate(zip(winner_idx, loser_idx)):
            W[winner] += lam[idx] / 2
            W[loser] += lam[idx] / 2

    A = K_inv + torch.diag(W)
    f_cov = torch.linalg.inv(
        A + jitter * torch.eye(n, dtype=K.dtype, device=K.device)
    )
    return f_map, f_cov


class GPPBOAgent(PBOAgent):
    """Laplace GP-PBO baseline for a one-dimensional discrete grid."""

    def __init__(
        self,
        lengthscale: float = 0.2,
        outputscale: float = 1.0,
        n_ts_samples: int = 2,
        support: str = "grid",
    ):
        if support != "grid":
            raise ValueError(
                "GPPBOAgent supports only support='grid'; "
                f"got {support!r}."
            )
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.n_ts_samples = n_ts_samples
        self.support = support

    def _pool_indices(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[list[int], list[int]]:
        """Map comparison points to their nearest candidate-grid indices."""
        grid = _one_dimensional_grid(candidate_pool)

        def snap(point: Point) -> int:
            return int((grid - _scalar_point(point, like=grid)).abs().argmin().item())

        winner_idx = [snap(winner) for winner, _ in comparisons]
        loser_idx = [snap(loser) for _, loser in comparisons]
        return winner_idx, loser_idx

    def _posterior(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[Tensor, Tensor]:
        grid = _one_dimensional_grid(candidate_pool)
        winner_idx, loser_idx = self._pool_indices(comparisons, grid)
        return _laplace_posterior(
            grid,
            winner_idx,
            loser_idx,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
        )

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> float:
        _one_dimensional_grid(candidate_pool)
        if not comparisons:
            return candidate_pool[candidate_pool.shape[0] // 2].item()
        f_mean, _ = self._posterior(comparisons, candidate_pool)
        return candidate_pool[f_mean.argmax()].item()

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: Tensor,
    ) -> tuple[float, float]:
        _one_dimensional_grid(candidate_pool)
        f_mean, f_cov = self._posterior(comparisons, candidate_pool)

        eye = torch.eye(len(f_mean), dtype=f_cov.dtype, device=f_cov.device)
        cov = (f_cov + f_cov.T) / 2 + 1e-4 * eye
        dist = torch.distributions.MultivariateNormal(
            f_mean,
            covariance_matrix=cov,
        )
        argmaxes = [
            candidate_pool[dist.sample().argmax()].item()
            for _ in range(self.n_ts_samples)
        ]

        if len(set(argmaxes)) == 1:
            challenger_idx = torch.randint(len(candidate_pool), (1,)).item()
            return argmaxes[0], candidate_pool[challenger_idx].item()
        return argmaxes[0], argmaxes[1]
