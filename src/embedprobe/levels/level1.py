"""Level 1 — retrieval performance.

For each source sentence, rank all target sentences by cosine similarity and
locate the true counterpart. Ranks are computed on the full similarity matrix,
so Recall@k, MRR and the CMC curve are exact for any k.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_KS = (1, 5, 10)


def true_ranks(sim_matrix: np.ndarray) -> np.ndarray:
    """Rank (1-based) of the true target for each source row."""
    sim_matrix = np.asarray(sim_matrix)
    n = sim_matrix.shape[0]
    diag = sim_matrix[np.arange(n), np.arange(n)]
    # Rank = 1 + number of candidates scoring strictly higher than the true one.
    return (sim_matrix > diag[:, None]).sum(axis=1) + 1


def retrieval_metrics(
    sim_matrix: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
    cmc_max_rank: int = 10,
) -> dict:
    sim_matrix = np.asarray(sim_matrix)
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("Level 1 expects a square src-x-tgt similarity matrix")

    ranks = true_ranks(sim_matrix)
    n = len(ranks)

    metrics = {"mrr": float((1.0 / ranks).mean()), "n_queries": int(n)}
    for k in ks:
        recall = float((ranks <= k).mean())
        metrics[f"recall@{k}"] = recall
        metrics[f"precision@{k}"] = recall / k

    cmc_ranks = list(range(1, cmc_max_rank + 1))
    cmc_values = [float((ranks <= k).mean()) for k in cmc_ranks]

    data = {
        "ranks": ranks.tolist(),
        "cmc_ranks": cmc_ranks,
        "cmc_values": cmc_values,
        "similarity_histogram": _histogram(sim_matrix),
    }
    return {"metrics": metrics, "data": data}


def _histogram(sim_matrix: np.ndarray, bins: int = 20) -> dict:
    counts, edges = np.histogram(sim_matrix.ravel(), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}
