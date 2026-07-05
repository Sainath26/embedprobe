"""Level 3 — retrieval-miss error taxonomy (the headline diagnostic).

A *miss* is a query whose top-1 retrieved target is not the true counterpart.
Each miss is classified by the token overlap between the retrieved and the true
target sentence (see :mod:`embedprobe.taxonomy`), and split into *near* misses
(true target ranked 2-3) and *far* misses (rank > 3).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from embedprobe.levels.level1 import true_ranks
from embedprobe.taxonomy import (
    DEFAULT_HIGH_THRESH,
    DEFAULT_LOW_THRESH,
    ERROR_TYPES,
    classify_miss,
    jaccard,
    levenshtein_ratio,
)

NEAR_MAX_RANK = 3


def error_taxonomy(
    sim_matrix: np.ndarray,
    src_texts: Sequence[str],
    tgt_texts: Sequence[str],
    topics: Optional[Sequence[str]] = None,
    high_thresh: float = DEFAULT_HIGH_THRESH,
    low_thresh: float = DEFAULT_LOW_THRESH,
    method: str = "jaccard",
) -> dict:
    sim_matrix = np.asarray(sim_matrix)
    n = sim_matrix.shape[0]
    if not (len(src_texts) == len(tgt_texts) == n):
        raise ValueError("src_texts and tgt_texts must match the similarity matrix size")

    ranks = true_ranks(sim_matrix)
    top1 = sim_matrix.argmax(axis=1)

    misses = []
    for i in range(n):
        if top1[i] == i:
            continue
        retrieved = tgt_texts[top1[i]]
        score, error_type = classify_miss(
            tgt_texts[i], retrieved, high_thresh, low_thresh, method
        )
        jaccard_score = jaccard(tgt_texts[i], retrieved)
        levenshtein_score = levenshtein_ratio(tgt_texts[i], retrieved)
        miss = {
            "query_index": int(i),
            "query_text": str(src_texts[i]),
            "true_target": str(tgt_texts[i]),
            "retrieved_target": str(retrieved),
            "retrieved_index": int(top1[i]),
            "true_rank": int(ranks[i]),
            "gold_rank": int(ranks[i]),
            "distance": "near" if ranks[i] <= NEAR_MAX_RANK else "far",
            "overlap_score": float(score),
            "jaccard": float(jaccard_score),
            "levenshtein": float(levenshtein_score),
            "error_type": error_type,
        }
        if topics is not None:
            miss["query_topic"] = str(topics[i])
            miss["retrieved_topic"] = str(topics[top1[i]])
        misses.append(miss)

    n_misses = len(misses)
    metrics = {
        "n_queries": int(n),
        "n_misses": n_misses,
        "miss_rate": n_misses / n if n else 0.0,
    }
    for et in ERROR_TYPES:
        count = sum(1 for m in misses if m["error_type"] == et)
        metrics[f"count::{et}"] = count
        metrics[f"pct::{et}"] = 100.0 * count / n_misses if n_misses else 0.0
    for dist in ("near", "far"):
        metrics[f"count::{dist}_misses"] = sum(1 for m in misses if m["distance"] == dist)

    return {
        "metrics": metrics,
        "data": {
            "misses": misses,
            "thresholds": {"high": high_thresh, "low": low_thresh, "method": method},
        },
    }
