"""Level 2 — topic-level structure and cohesion.

Uses topic labels to check whether cross-lingual retrieval respects topic
boundaries (retrieval-based topic confusion) and how similar topics are to each
other on average (topic-pair average cosine). Optionally computes 2-D UMAP
coordinates of source/target embeddings for visualization.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def topic_structure(
    sim_matrix: np.ndarray,
    topics: Sequence[str],
    src_embeddings: Optional[np.ndarray] = None,
    tgt_embeddings: Optional[np.ndarray] = None,
    umap_seed: int = 42,
) -> dict:
    sim_matrix = np.asarray(sim_matrix)
    topics = [str(t) for t in topics]
    if len(topics) != sim_matrix.shape[0]:
        raise ValueError("topics must have one label per row of the similarity matrix")

    unique_topics = sorted(set(topics))
    predicted = [topics[j] for j in sim_matrix.argmax(axis=1)]

    cm_recall = confusion_matrix(topics, predicted, labels=unique_topics, normalize="true")
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_precision = confusion_matrix(topics, predicted, labels=unique_topics, normalize="pred")
    cm_precision = np.nan_to_num(cm_precision)

    metrics = {"topic_retrieval_accuracy": float(np.mean(np.array(topics) == np.array(predicted)))}
    for idx, t in enumerate(unique_topics):
        metrics[f"recall::{t}"] = float(cm_recall[idx, idx])
        metrics[f"precision::{t}"] = float(cm_precision[idx, idx])

    # Topic-pair average cosine: how separable topics are in the shared space.
    topic_arr = np.array(topics)
    n_topics = len(unique_topics)
    avg_cosine = np.zeros((n_topics, n_topics))
    for i, ti in enumerate(unique_topics):
        rows = topic_arr == ti
        for j, tj in enumerate(unique_topics):
            cols = topic_arr == tj
            avg_cosine[i, j] = float(sim_matrix[np.ix_(rows, cols)].mean())

    on_diag = float(np.mean(np.diag(avg_cosine)))
    off_mask = ~np.eye(n_topics, dtype=bool)
    off_diag = float(avg_cosine[off_mask].mean()) if n_topics > 1 else 0.0
    metrics["topic_cohesion"] = on_diag
    metrics["topic_leakage"] = off_diag

    data = {
        "topics": unique_topics,
        "confusion_recall": cm_recall.tolist(),
        "confusion_precision": cm_precision.tolist(),
        "topic_avg_cosine": avg_cosine.tolist(),
    }

    if src_embeddings is not None and tgt_embeddings is not None:
        coords = _umap_coords(src_embeddings, tgt_embeddings, umap_seed)
        if coords is not None:
            data["umap"] = coords

    return {"metrics": metrics, "data": data}


def _umap_coords(src_emb: np.ndarray, tgt_emb: np.ndarray, seed: int) -> Optional[dict]:
    """2-D UMAP projection of both languages jointly; None if umap-learn is absent."""
    try:
        import umap
    except ImportError:
        return None

    stacked = np.vstack([src_emb, tgt_emb])
    reducer = umap.UMAP(metric="cosine", random_state=seed)
    coords = reducer.fit_transform(stacked)
    n = len(src_emb)
    return {"src": coords[:n].tolist(), "tgt": coords[n:].tolist()}
