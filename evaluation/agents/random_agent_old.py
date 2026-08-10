"""Legacy random baseline agent."""

from __future__ import annotations

import random
from collections import Counter

import torch

from .base import Comparison, PBOAgent, Point, candidate_value


class RandomAgentOld(PBOAgent):
    """Random-pair baseline kept for comparison with the current random agent."""

    def __init__(self, seed: int = 0, support: str = "grid"):
        if support not in {"grid", "continuous_rff"}:
            raise ValueError(f"Unknown RandomAgentOld support {support!r}.")
        self._rng = random.Random(seed)
        self.support = support

    def _continuous_random_point(self, candidate_pool: torch.Tensor) -> Point:
        candidates = torch.as_tensor(candidate_pool)
        input_dim = (
            1
            if candidates.ndim == 1
            else candidates.reshape(candidates.shape[0], -1).shape[-1]
        )
        if input_dim == 1:
            return self._rng.random()
        return tuple(self._rng.random() for _ in range(input_dim))

    def suggest_pair(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> tuple[Point, Point]:
        if self.support == "grid":
            if candidate_pool.ndim == 1:
                pool = candidate_pool.tolist()
                x1, x2 = self._rng.sample(pool, 2)
                return x1, x2
            idx1, idx2 = self._rng.sample(range(len(candidate_pool)), 2)
            return (
                candidate_value(candidate_pool[idx1]),
                candidate_value(candidate_pool[idx2]),
            )

        if not comparisons:
            return (
                self._continuous_random_point(candidate_pool),
                self._continuous_random_point(candidate_pool),
            )
        incumbent = comparisons[-1][0]
        return incumbent, self._continuous_random_point(candidate_pool)

    def recommend(
        self,
        comparisons: list[Comparison],
        candidate_pool: torch.Tensor,
    ) -> Point:
        if self.support == "grid":
            if not comparisons:
                if candidate_pool.ndim == 1:
                    return self._rng.choice(candidate_pool.tolist())
                idx = self._rng.choice(range(len(candidate_pool)))
                return candidate_value(candidate_pool[idx])
            wins = Counter(w for w, _ in comparisons)
            return max(wins, key=wins.get)

        if comparisons:
            return comparisons[-1][0]
        return self._continuous_random_point(candidate_pool)
