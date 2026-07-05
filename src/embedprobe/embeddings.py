"""Embedding extraction and similarity computation.

Heavy dependencies (sentence-transformers / torch) are imported lazily so that
the pure-numpy diagnostics in :mod:`embedprobe.levels` stay usable without them.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model: Union[str, object]):
    """Return a SentenceTransformer, loading it from the Hub if given a name."""
    if isinstance(model, str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model)
    return model


def encode(
    model,
    texts: Sequence[str],
    batch_size: int = 32,
    device: Optional[str] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts with a (loaded) SentenceTransformer into a 2-D array."""
    return model.encode(
        [str(t) for t in texts],
        batch_size=batch_size,
        device=device,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def pairwise_cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix between two sets of embeddings."""
    return cosine_similarity(emb_a, emb_b)


def model_info(model, fallback_name: str = "") -> dict:
    """Best-effort architecture metadata for a SentenceTransformer."""
    info = {
        "name": getattr(model, "model_name_or_path", None) or fallback_name,
        "hub_id": getattr(model, "model_name_or_path", None) or fallback_name,
        "embedding_dim": None,
        "arch": None,
        "layers": None,
        "vocab_size": None,
        "params": None,
        "license": None,
    }
    try:
        info["embedding_dim"] = int(model.get_sentence_embedding_dimension())
    except Exception:
        pass
    try:
        cfg = model._first_module().auto_model.config
        info["arch"] = getattr(cfg, "model_type", None)
        for key in ("num_hidden_layers", "num_layers", "n_layer"):
            if hasattr(cfg, key):
                info["layers"] = int(getattr(cfg, key))
                break
        info["vocab_size"] = getattr(cfg, "vocab_size", None)
        info["license"] = getattr(cfg, "license", None)
        try:
            info["params"] = int(model._first_module().auto_model.num_parameters())
        except Exception:
            pass
    except Exception:
        pass
    return info
