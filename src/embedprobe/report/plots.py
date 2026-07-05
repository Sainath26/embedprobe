"""Matplotlib renderings of level data, encoded as base64 <img> tags."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from embedprobe.taxonomy import ERROR_TYPES


def fig_to_img(fig, thumb_width: int = 220) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf8")
    src = f"data:image/png;base64,{b64}"
    return (
        f'<img src="{src}" width="{thumb_width}px" '
        f'style="cursor:pointer;" onclick="openModal(this.src)"/>'
    )


def signal_noise_plot(level0: dict) -> str:
    data = level0["data"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(data["random_sims"], bins=50, alpha=0.5, label="Random pairs")
    ax.hist(data["true_sims"], bins=50, alpha=0.5, label="True pairs")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("True vs. random pair similarity")
    ax.legend()
    return fig_to_img(fig)


def cmc_plot(level1: dict) -> str:
    data = level1["data"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(data["cmc_ranks"], data["cmc_values"], marker="o")
    ax.set_xlabel("Rank threshold k")
    ax.set_ylabel("Fraction of queries with rank <= k")
    ax.set_title("CMC curve")
    ax.set_ylim(0, 1.05)
    return fig_to_img(fig)


def similarity_hist_plot(level1: dict) -> str:
    hist = level1["data"]["similarity_histogram"]
    edges = np.array(hist["bin_edges"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(edges[:-1], hist["counts"], width=np.diff(edges), align="edge")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Similarity score distribution")
    return fig_to_img(fig)


def _heatmap(matrix, labels, title, cmap="viridis") -> str:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig_to_img(fig)


def topic_confusion_plot(level2: dict) -> str:
    data = level2["data"]
    return _heatmap(data["confusion_recall"], data["topics"], "Topic confusion (recall)", "Blues")


def topic_cosine_plot(level2: dict) -> str:
    data = level2["data"]
    return _heatmap(data["topic_avg_cosine"], data["topics"], "Topic-pair avg cosine")


def umap_plot(level2: dict) -> str:
    umap_data = level2["data"].get("umap")
    if not umap_data:
        return ""
    src = np.asarray(umap_data["src"])
    tgt = np.asarray(umap_data["tgt"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(src[:, 0], src[:, 1], marker="*", s=30, alpha=0.7, label="source")
    ax.scatter(tgt[:, 0], tgt[:, 1], marker="d", s=30, alpha=0.7, label="target")
    ax.set_title("UMAP projection (both languages)")
    ax.legend()
    return fig_to_img(fig)


def error_types_plot(level3: dict) -> str:
    metrics = level3["metrics"]
    counts = {et: metrics.get(f"count::{et}", 0) for et in ERROR_TYPES}
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(range(len(counts)), list(counts.values()))
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([et.replace("_", "\n") for et in counts], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Retrieval-miss error types")
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")
    return fig_to_img(fig)
