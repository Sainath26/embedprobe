"""Level 0 — signal-to-noise separability.

Compares cosine similarities of true (diagonal) pairs against a matched random
sample of off-diagonal pairs. A model that has not learned cross-lingual
alignment shows heavily overlapping distributions and a KS p-value near 1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import ks_2samp


def signal_to_noise(sim_matrix: np.ndarray, seed: Optional[int] = None) -> dict:
    sim_matrix = np.asarray(sim_matrix)
    n = sim_matrix.shape[0]
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("Level 0 expects a square src-x-tgt similarity matrix")
    if n < 2:
        raise ValueError("Need at least 2 pairs for signal-to-noise analysis")

    true_sims = np.diag(sim_matrix).astype(float)
    off_diag = sim_matrix[~np.eye(n, dtype=bool)].astype(float)

    rng = np.random.RandomState(seed)
    random_sims = rng.choice(off_diag, size=n, replace=len(off_diag) < n)

    ks = ks_2samp(true_sims, random_sims)
    random_std = float(random_sims.std())
    snr = float((true_sims.mean() - random_sims.mean()) / random_std) if random_std > 0 else float("inf")
    overlap = _distribution_overlap(true_sims, random_sims)

    metrics = {
        "mean_true": float(true_sims.mean()),
        "std_true": float(true_sims.std()),
        "mean_random": float(random_sims.mean()),
        "std_random": random_std,
        "snr": snr,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "p_value": float(ks.pvalue),
        "overlap_fraction": overlap,
        "verdict": _verdict(snr, float(ks.pvalue), overlap),
        "n_pairs": int(n),
    }
    data = {
        "true_sims": true_sims.tolist(),
        "random_sims": random_sims.tolist(),
    }
    return {"metrics": metrics, "data": data}


def _distribution_overlap(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if lo == hi:
        return 1.0
    counts_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    counts_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    overlap = np.minimum(counts_a, counts_b) * widths
    return float(np.clip(overlap.sum(), 0.0, 1.0))


def _verdict(snr: float, p_value: float, overlap: float) -> str:
    if p_value < 0.01 and snr >= 2.0 and overlap <= 0.25:
        return "strong signal"
    if p_value < 0.05 and snr > 0.5:
        return "weak signal"
    return "poor separation"
