from dataclasses import dataclass
from functools import partial

import torch

from evaluation.agents.qeubo_agent import QEUBOAgent
from pfns.priors import Batch
from pfns.priors.pref.pref_gp_1d_qeubo import sample_gp_batch
from pfns.priors.prior import PriorConfig


def get_batch(
    batch_size=2,
    seq_len=100,
    num_features=2,
    hyperparameters=None,
    device="cpu",
    single_eval_pos=None,
    *,
    lengthscale=0.2,
    outputscale=1.0,
    mean_constant=0.0,
    noise_std=0.05,
    jitter=1e-6,
    pool_size=100,
    n_init=5,
    **kwargs,
):
    """Генерирует preference-задачи на qEUBO-траекториях из фиксированного pool."""
    assert num_features >= 2 and num_features % 2 == 0
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 2
    assert n_init >= 0

    gp_dim = num_features // 2
    pool_x = torch.rand(batch_size, pool_size, gp_dim, device=device)
    pool_f, _ = sample_gp_batch(
        pool_x,
        lengthscale=lengthscale,
        outputscale=outputscale,
        mean_constant=mean_constant,
        noise_std=None,
        jitter=jitter,
    )

    x = torch.zeros(
        batch_size,
        seq_len,
        num_features,
        dtype=pool_x.dtype,
        device=device,
    )
    target = torch.zeros(
        batch_size,
        seq_len,
        dtype=pool_f.dtype,
        device=device,
    )
    comparisons = torch.empty(batch_size, 0, 2, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)

    agent = QEUBOAgent(
        fit_hyperparams=False,
        device=device,
        support="grid",
        gp_lengthscale=lengthscale,
        gp_outputscale=outputscale,
    )

    for t in range(single_eval_pos):
        if t < n_init:
            pair_indices = torch.rand(
                batch_size,
                pool_size,
                device=device,
            ).topk(2, dim=-1).indices
        else:
            pair_indices = agent.suggest_pairs_batched_grid(pool_x, comparisons)

        observed = pool_f.gather(1, pair_indices)
        if noise_std > 0:
            observed = observed + noise_std * torch.randn_like(observed)
        prefer_second = observed[:, 1] > observed[:, 0]
        ordered_indices = pair_indices.clone()
        ordered_indices[prefer_second] = ordered_indices[prefer_second].flip(-1)
        comparisons = torch.cat(
            [comparisons, ordered_indices.unsqueeze(1)],
            dim=1,
        )
        pair = pool_x[batch_indices, ordered_indices]
        x[:, t] = pair.reshape(batch_size, num_features)

    num_queries = seq_len - single_eval_pos
    if num_queries > 0:
        # The next qEUBO pair also provides its posterior-mean incumbent.
        next_qeubo_pair = agent.suggest_pairs_batched_grid(pool_x, comparisons)
        incumbent_indices = next_qeubo_pair[:, 0]

        pair_indices = torch.randint(
            pool_size,
            (batch_size, num_queries, 2),
            device=device,
        )
        query_type = torch.randint(
            5,
            (batch_size, num_queries),
            device=device,
        )

        # Three of five types train the exact identity max(f(x), f(x)) = f(x).
        diagonal = query_type < 3
        pair_indices[..., 1][diagonal] = pair_indices[..., 0][diagonal]

        # One type matches the incumbent-versus-challenger region used by qEUBO.
        incumbent_duel = query_type == 3
        incumbent_grid = incumbent_indices.unsqueeze(-1).expand(-1, num_queries)
        pair_indices[..., 0][incumbent_duel] = incumbent_grid[incumbent_duel]

        # Acquisition queries must contain two distinct points.
        non_diagonal = ~diagonal
        repeated = non_diagonal & (pair_indices[..., 0] == pair_indices[..., 1])
        pair_indices[..., 1][repeated] += 1
        pair_indices[..., 1].remainder_(pool_size)

        # Replace one incumbent duel with the exact qEUBO proposal when available.
        random_priority = torch.rand(
            batch_size,
            num_queries,
            device=device,
        ).masked_fill(~incumbent_duel, -1.0)
        qeubo_slots = random_priority.argmax(dim=-1)
        has_incumbent_duel = incumbent_duel.any(dim=-1)
        qeubo_batches = batch_indices.squeeze(-1)[has_incumbent_duel]
        pair_indices[
            qeubo_batches,
            qeubo_slots[has_incumbent_duel],
        ] = next_qeubo_pair[has_incumbent_duel]

        query_batch_indices = torch.arange(batch_size, device=device)[:, None, None]
        query_x = pool_x[query_batch_indices, pair_indices]
        query_f = pool_f[query_batch_indices, pair_indices]
        x[:, single_eval_pos:] = query_x.reshape(
            batch_size,
            num_queries,
            num_features,
        )
        target[:, single_eval_pos:] = query_f.max(dim=-1).values

    return Batch(
        x=x,
        y=target,
        target_y=target,
        single_eval_pos=single_eval_pos,
    )


@dataclass(frozen=True)
class PrefGPqEUBOPolicyAlignedPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    pool_size: int = 100
    n_init: int = 5

    def create_get_batch_method(self):
        return partial(
            get_batch,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            mean_constant=self.mean_constant,
            noise_std=self.noise_std,
            jitter=self.jitter,
            pool_size=self.pool_size,
            n_init=self.n_init,
        )
