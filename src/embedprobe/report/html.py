"""Self-contained HTML diagnostic report."""

from __future__ import annotations

import html as html_lib

from embedprobe.report import plots
from embedprobe.taxonomy import ERROR_TYPES

_PAGE_STYLE = """
<style>
  body { font-family: system-ui, sans-serif; margin: 2em; color: #222; }
  h1 { text-align: center; }
  table.summary { border-collapse: collapse; margin: 1em auto; font-size: 0.85em; }
  table.summary th, table.summary td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; }
  table.summary th { background: #f5f5f5; }
  .model-section { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0; }
  .plot-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
  .diagnosis { background: #f8f9fb; border-left: 4px solid #4a7cc9; padding: 0.6em 1em; margin: 0.8em 0; }
  .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
           background: rgba(0,0,0,0.8); justify-content: center; align-items: center; z-index: 9999; }
  .modal img { max-width: 90%; max-height: 90%; }
</style>
<script>
  function sortTable(table, col) {
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const current = table.dataset.sortCol === String(col) && table.dataset.sortDir === 'asc';
    const dir = current ? -1 : 1;
    rows.sort((a, b) => {
      const av = a.cells[col].innerText.trim();
      const bv = b.cells[col].innerText.trim();
      const an = Number(av);
      const bn = Number(bv);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
      return av.localeCompare(bv) * dir;
    });
    rows.forEach(row => tbody.appendChild(row));
    table.dataset.sortCol = String(col);
    table.dataset.sortDir = dir === 1 ? 'asc' : 'desc';
  }
  function makeSummarySortable() {
    document.querySelectorAll('table.summary').forEach(table => {
      Array.from(table.tHead.rows[0].cells).forEach((cell, idx) => {
        cell.style.cursor = 'pointer';
        cell.title = 'Sort';
        cell.onclick = () => sortTable(table, idx);
      });
    });
  }
  function openModal(src) {
    const m = document.getElementById('imgModal');
    m.querySelector('img').src = src;
    m.style.display = 'flex';
  }
  document.addEventListener('DOMContentLoaded', makeSummarySortable);
</script>
<div id="imgModal" class="modal" onclick="this.style.display='none'"><img/></div>
"""


def render_html(report) -> str:
    parts = [_PAGE_STYLE]
    run = report.run_meta
    pair = f"{run.get('src_col', 'source')} → {run.get('tgt_col', 'target')}"
    parts.append(f"<h1>embedprobe diagnostic report — {html_lib.escape(pair)}</h1>")
    if run:
        n = run.get("n_pairs", "?")
        parts.append(f"<p style='text-align:center;'>{n} pairs · seed {run.get('seed', '—')} · embedprobe {run.get('version', '')}</p>")

    summary = report.summary()
    if not summary.empty:
        parts.append("<h2>Cross-model summary</h2>")
        parts.append(summary.round(4).to_html(classes="summary", border=0))

    for diag in report.models:
        parts.append(_model_section(diag))

    return "\n".join(parts)


def _model_section(diag) -> str:
    name = html_lib.escape(diag.model_name)
    section = [f'<div class="model-section"><h2>{name}</h2>']

    if diag.meta:
        meta_bits = " · ".join(
            f"{k}: {html_lib.escape(str(v))}" for k, v in diag.meta.items() if v is not None
        )
        section.append(f"<p>{meta_bits}</p>")

    section.append(f'<div class="diagnosis">{html_lib.escape(_diagnosis(diag))}</div>')

    imgs = []
    if diag.level0:
        imgs.append(plots.signal_noise_plot(diag.level0))
    if diag.level1:
        imgs.append(plots.cmc_plot(diag.level1))
        imgs.append(plots.similarity_hist_plot(diag.level1))
    if diag.level2:
        imgs.append(plots.topic_confusion_plot(diag.level2))
        imgs.append(plots.topic_cosine_plot(diag.level2))
        umap_img = plots.umap_plot(diag.level2)
        if umap_img:
            imgs.append(umap_img)
    if diag.level3:
        imgs.append(plots.error_types_plot(diag.level3))
    section.append('<div class="plot-row">' + "".join(imgs) + "</div>")

    section.append("</div>")
    return "\n".join(section)


def _diagnosis(diag) -> str:
    """One-sentence plain-language reading of the metrics."""
    bits = []
    if diag.level0:
        m = diag.level0["metrics"]
        sep = "cleanly separates" if m["ks_p_value"] < 0.01 and m["snr"] > 2 else "struggles to separate"
        bits.append(f"{sep} true pairs from noise (SNR {m['snr']:.1f})")
    if diag.level1:
        m = diag.level1["metrics"]
        r1 = m.get("recall@1")
        if r1 is not None:
            bits.append(f"retrieves the true counterpart first {100 * r1:.0f}% of the time (MRR {m['mrr']:.2f})")
    if diag.level3:
        m = diag.level3["metrics"]
        if m["n_misses"] > 0:
            dominant = max(ERROR_TYPES, key=lambda et: m.get(f"count::{et}", 0))
            bits.append(
                f"misses are mostly {dominant.replace('_', ' ')} "
                f"({m.get(f'pct::{dominant}', 0):.0f}% of {m['n_misses']} misses)"
            )
        else:
            bits.append("no retrieval misses on this dataset")
    return "This model " + "; ".join(bits) + "." if bits else "No diagnostics computed."
