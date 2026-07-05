"""Loading and validating parallel-pair datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import pandas as pd

TOPIC_COL = "topic"


def load_pairs(
    source: Union[str, Path, pd.DataFrame],
    src_col: str,
    tgt_col: str,
    topic_col: Optional[str] = TOPIC_COL,
    max_pairs: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load a parallel dataset from a DataFrame, CSV or Parquet file.

    Returns a DataFrame with the source, target and (if present) topic columns,
    with empty/missing texts dropped and the index reset. When ``max_pairs`` is
    given, rows are sampled reproducibly with ``seed``.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

    missing = [c for c in (src_col, tgt_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s) {missing}; found {list(df.columns)}"
        )

    cols = [src_col, tgt_col]
    if topic_col is not None and topic_col in df.columns:
        cols.append(topic_col)
    df = df[cols]

    for col in (src_col, tgt_col):
        df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]

    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=seed)

    return df.reset_index(drop=True)


def has_topics(df: pd.DataFrame, topic_col: str = TOPIC_COL) -> bool:
    return topic_col in df.columns and df[topic_col].notna().any()


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint of the loaded evaluation pairs."""
    normalized = df.astype("string").fillna("")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
