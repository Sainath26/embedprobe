# embedprobe

A diagnostic toolkit for evaluating and selecting language-model embedding spaces.

> **Status: early development.** This is a placeholder release to reserve the name.
> The first working version is coming soon.

## What it will do

`embedprobe` helps you choose the right embedding model for _your_ downstream task
by diagnosing _why_ models succeed or fail on _your_ data and language pairs, going
beyond aggregate leaderboard scores. It probes an embedding space across four levels:

- **Signal-to-noise separability** — how cleanly true pairs separate from noise.
- **Retrieval performance** — Recall@K, MRR, and cumulative-match analysis.
- **Topic-level structure** — UMAP projections and topic-cohesion heatmaps.
- **Error categorisation** — a taxonomy of retrieval misses (lexical / semantic / topic-boundary).

## Install

```bash
pip install embedprobe
```

## License

MIT
