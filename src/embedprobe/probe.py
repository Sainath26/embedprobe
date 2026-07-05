"""The main entry point: run all diagnostic levels for a set of models."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from embedprobe import __version__
from embedprobe.data import TOPIC_COL, dataset_hash, has_topics, load_pairs
from embedprobe.embeddings import encode, load_model, model_info, pairwise_cosine
from embedprobe.levels import error_taxonomy, retrieval_metrics, signal_to_noise, topic_structure
from embedprobe.report import ModelDiagnostics, ProbeReport
from embedprobe.taxonomy import DEFAULT_HIGH_THRESH, DEFAULT_LOW_THRESH

DEFAULT_LEVELS = (0, 1, 2, 3)


def probe(
    models: Union[str, object, Sequence],
    data: Union[str, pd.DataFrame],
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
    topic_col: str = TOPIC_COL,
    levels: Sequence[int] = DEFAULT_LEVELS,
    seed: Optional[int] = 42,
    batch_size: int = 32,
    device: Optional[str] = None,
    max_pairs: Optional[int] = None,
    high_thresh: float = DEFAULT_HIGH_THRESH,
    low_thresh: float = DEFAULT_LOW_THRESH,
    overlap_method: str = "jaccard",
    compute_umap: bool = False,
    show_progress: bool = True,
) -> ProbeReport:
    """Run the requested diagnostic levels for each model and bundle the results.

    ``models`` may be Hub model names, loaded SentenceTransformers, or a mix.
    ``data`` may be a DataFrame or a path to a CSV/Parquet file with one row per
    parallel pair (``src_col``, ``tgt_col`` and optionally ``topic_col``).
    """
    src_col = src_col or src_lang
    tgt_col = tgt_col or tgt_lang
    if src_col is None or tgt_col is None:
        raise ValueError("probe requires src_col/tgt_col or src_lang/tgt_lang column names")

    if isinstance(models, (str,)) or not isinstance(models, (list, tuple)):
        models = [models]
    levels = set(levels)

    df = load_pairs(data, src_col, tgt_col, topic_col, max_pairs=max_pairs, seed=seed)
    data_hash = dataset_hash(df)
    src_texts = df[src_col].astype(str).tolist()
    tgt_texts = df[tgt_col].astype(str).tolist()
    topics = df[topic_col].astype(str).tolist() if has_topics(df, topic_col) else None

    diagnostics = []
    for entry in models:
        name = entry if isinstance(entry, str) else getattr(entry, "model_name_or_path", str(entry))
        model = load_model(entry)

        src_emb = encode(model, src_texts, batch_size, device, show_progress)
        tgt_emb = encode(model, tgt_texts, batch_size, device, show_progress)
        sim_matrix = pairwise_cosine(src_emb, tgt_emb)

        meta = model_info(model, name)
        meta.update({"embedprobe_version": __version__, "dataset_hash": data_hash, "seed": seed})
        diag = ModelDiagnostics(model_name=name, meta=meta)
        if 0 in levels:
            diag.level0 = signal_to_noise(sim_matrix, seed=seed)
        if 1 in levels:
            diag.level1 = retrieval_metrics(sim_matrix)
        if 2 in levels and topics is not None:
            diag.level2 = topic_structure(
                sim_matrix,
                topics,
                src_embeddings=src_emb if compute_umap else None,
                tgt_embeddings=tgt_emb if compute_umap else None,
            )
        if 3 in levels:
            diag.level3 = error_taxonomy(
                sim_matrix, src_texts, tgt_texts, topics,
                high_thresh=high_thresh, low_thresh=low_thresh, method=overlap_method,
            )
        diagnostics.append(diag)

    run_meta = {
        "version": __version__,
        "src_col": src_col,
        "tgt_col": tgt_col,
        "n_pairs": len(df),
        "seed": seed,
        "dataset_hash": data_hash,
        "levels": sorted(levels),
        "has_topics": topics is not None,
    }
    return ProbeReport(models=diagnostics, run_meta=run_meta)
