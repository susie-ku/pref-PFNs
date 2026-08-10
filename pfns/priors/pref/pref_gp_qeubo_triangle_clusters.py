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
    """Генерирует context-треугольники и query для их недостающих рёбер."""
    assert num_features >= 2 and num_features % 2 == 0
    assert single_eval_pos is not None
    assert 0 <= single_eval_pos <= seq_len
    assert pool_size >= 3
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
    missing_triangle_pairs = []
    triangle_anchor = None
    triangle_second = None
    triangle_third = None

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
        elif (t - n_init) % 2 == 0:
            pair_indices = agent.suggest_pairs_batched_grid(pool_x, comparisons)
            triangle_anchor = pair_indices[:, 0]
            triangle_second = pair_indices[:, 1]

            low = torch.minimum(triangle_anchor, triangle_second)
            high = torch.maximum(triangle_anchor, triangle_second)
            triangle_third = torch.randint(
                pool_size - 2,
                (batch_size,),
                device=device,
            )
            triangle_third += (triangle_third >= low).long()
            triangle_third += (triangle_third >= high).long()
        else:
            pair_indices = torch.stack([triangle_anchor, triangle_third], dim=-1)
            missing_triangle_pairs.append(
                torch.stack([triangle_second, triangle_third], dim=-1)
            )

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
        pair_indices = torch.randint(
            pool_size,
            (batch_size, num_queries, 2),
            device=device,
        )
        repeated = pair_indices[..., 0] == pair_indices[..., 1]
        pair_indices[..., 1][repeated] += 1
        pair_indices[..., 1].remainder_(pool_size)

        query_type = torch.arange(num_queries, device=device) % 3
        triangle_queries = query_type == 0
        diagonal_queries = query_type == 1
        pair_indices[:, diagonal_queries, 1] = pair_indices[:, diagonal_queries, 0]

        if missing_triangle_pairs:
            triangles = torch.stack(missing_triangle_pairs, dim=1)
            triangle_slots = torch.randint(
                triangles.shape[1],
                (batch_size, num_queries),
                device=device,
            )
            selected_triangles = triangles.gather(
                1,
                triangle_slots.unsqueeze(-1).expand(-1, -1, 2),
            )
            pair_indices[:, triangle_queries] = selected_triangles[:, triangle_queries]

        global_slots = torch.where(query_type == 2)[0]
        if len(global_slots) > 0:
            next_qeubo_pair = agent.suggest_pairs_batched_grid(pool_x, comparisons)
            pair_indices[:, global_slots[0]] = next_qeubo_pair

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
class PrefGPqEUBOTriangleClustersPriorConfig(PriorConfig):
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
