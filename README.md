# embedprobe

**A diagnostic toolkit for evaluating and selecting multilingual embedding models on _your_ data.**

Leaderboards like [MTEB](https://github.com/embeddings-benchmark/mteb) tell you _which_ embedding
model ranks higher on average. **embedprobe tells you _why_ a model fails on your domain and
language pair, ** so you can pick a compact encoder for your task without fine-tuning every
candidate.

Given a parallel dataset (source text, target text, topic), embedprobe dissects each candidate
model across four diagnostic levels:

| Level                    | Question it answers                                         | Signals                                                                                                                     |
| ------------------------ | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **0 — Signal vs. noise** | Does the model separate true pairs from random ones at all? | true-vs-random cosine distributions, SNR, Kolmogorov–Smirnov test                                                           |
| **1 — Retrieval**        | How reliably does it retrieve the true counterpart?         | Recall@k, Precision@k, MRR, CMC curve                                                                                       |
| **2 — Topic structure**  | Is the space organized by meaning or leaking across topics? | retrieval-based topic confusion, topic-pair average cosine, UMAP projections                                                |
| **3 — Error taxonomy**   | _Why_ do retrievals miss?                                   | Jaccard-based categorization of misses into **lexical confusion**, **semantic confusion**, and **topic-boundary fuzziness** |

The Level 3 error taxonomy is the headline: instead of a single aggregate score, each retrieval
miss is classified by token-overlap between the retrieved and the true target, telling you whether
a model is being fooled by surface overlap, drifting semantically, or blurring topic boundaries.

## Install

```bash
pip install embedprobe
```

UMAP projection support is included for Level 2 visual diagnostics.

## Quickstart

```python
import pandas as pd
from embedprobe import probe

# parallel data: one row per pair, plus a topic column
df = pd.read_csv("pairs.csv")   # columns: en, es, topic

report = probe(
    models=["sentence-transformers/LaBSE",
            "sentence-transformers/distiluse-base-multilingual-cased-v2"],
    data=df,
    src_col="en",
    tgt_col="es",
    seed=42,
)

report.summary()            # cross-model DataFrame of all metrics
report.to_json("report.json")
report.to_html("report.html")   # self-contained diagnostic dashboard
```

Or from the command line:

```bash
embedprobe run --models sentence-transformers/LaBSE --data pairs.csv \
    --src en --tgt es --out report
```

## Status

Pre-release (`0.x`). The toolkit originates from an MSc dissertation study of 21
sentence-transformer models across EN–ES, EN–FR and EN–ZH parallel data; the packaged version is
being hardened for a 2026 workshop submission. APIs may change until `1.0`.

## Roadmap

- [ ] Empirical validation of the Jaccard taxonomy thresholds
- [ ] Selection-prediction study: do these diagnostics predict downstream task ranking?
- [ ] MTEB adapter: shortlist from the leaderboard → diagnose on your data
- [ ] Decoder-only LLM support (MEXA-style alignment probing)

## License

MIT
