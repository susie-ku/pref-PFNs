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
    query_cluster_size=8,
    local_fraction=0.5,
    local_neighbor_count=8,
    **kwargs,
):
    """Генерирует query-строки с общим anchor и ближайшими challengers."""
    assert num_features >= 2 and num_features % 2 == 0
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 2
    assert n_init >= 0
    assert query_cluster_size >= 2
    assert 0.0 <= local_fraction <= 1.0
    assert local_neighbor_count >= 1

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
        next_qeubo_pair = agent.suggest_pairs_batched_grid(pool_x, comparisons)
        num_clusters = (num_queries + query_cluster_size - 1) // query_cluster_size

        anchors = torch.randint(
            pool_size,
            (batch_size, num_clusters),
            device=device,
        )
        if comparisons.shape[1] > 0:
            winner_slots = torch.randint(
                comparisons.shape[1],
                (batch_size, num_clusters),
                device=device,
            )
            observed_winners = comparisons[..., 0].gather(1, winner_slots)
            use_observed = torch.arange(num_clusters, device=device) % 2 == 1
            anchors[:, use_observed] = observed_winners[:, use_observed]
        anchors[:, 0] = next_qeubo_pair[:, 0]

        anchor_x = pool_x[batch_indices, anchors]
        distances = torch.cdist(anchor_x, pool_x)
        distances.scatter_(2, anchors.unsqueeze(-1), torch.inf)
        nearest = distances.topk(
            min(local_neighbor_count, pool_size - 1),
            dim=-1,
            largest=False,
        ).indices

        query_positions = torch.arange(num_queries, device=device)
        cluster_ids = query_positions // query_cluster_size
        query_anchors = anchors[:, cluster_ids]
        challengers = torch.randint(
            pool_size,
            (batch_size, num_queries),
            device=device,
        )

        nearest_for_query = nearest[:, cluster_ids]
        neighbor_slots = torch.randint(
            nearest.shape[-1],
            (batch_size, num_queries, 1),
            device=device,
        )
        local_challengers = nearest_for_query.gather(2, neighbor_slots).squeeze(-1)
        positions_in_cluster = query_positions % query_cluster_size
        num_local = round((query_cluster_size - 1) * local_fraction)
        local_queries = (positions_in_cluster > 0) & (
            positions_in_cluster <= num_local
        )
        challengers[:, local_queries] = local_challengers[:, local_queries]

        repeated = challengers == query_anchors
        challengers[repeated] += 1
        challengers.remainder_(pool_size)
        pair_indices = torch.stack([query_anchors, challengers], dim=-1)

        diagonal = positions_in_cluster == 0
        pair_indices[:, diagonal, 1] = pair_indices[:, diagonal, 0]
        if num_queries > 1:
            pair_indices[:, 1] = next_qeubo_pair

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
class PrefGPqEUBOGeometricClustersPriorConfig(PriorConfig):
    lengthscale: float = 0.2
    outputscale: float = 1.0
    mean_constant: float = 0.0
    noise_std: float = 0.05
    jitter: float = 1e-6
    pool_size: int = 100
    n_init: int = 5
    query_cluster_size: int = 8
    local_fraction: float = 0.5
    local_neighbor_count: int = 8

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
            query_cluster_size=self.query_cluster_size,
            local_fraction=self.local_fraction,
            local_neighbor_count=self.local_neighbor_count,
        )
