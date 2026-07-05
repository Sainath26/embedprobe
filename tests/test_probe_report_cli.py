import json

import numpy as np
import pandas as pd

from embedprobe.cli import _normalize_models
from embedprobe.probe import probe


class FakeSentenceTransformer:
    model_name_or_path = "fake/model"

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, batch_size=32, device=None, show_progress_bar=False, convert_to_numpy=True):
        vectors = {
            "hello": [1.0, 0.0],
            "goodbye": [0.0, 1.0],
            "hola": [1.0, 0.0],
            "adios": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts], dtype=float)


def test_probe_runs_all_levels_with_fake_model_and_writes_report_surfaces(tmp_path):
    df = pd.DataFrame(
        {
            "en": ["hello", "goodbye"],
            "es": ["hola", "adios"],
            "topic": ["greeting", "farewell"],
        }
    )

    report = probe(
        models=FakeSentenceTransformer(),
        data=df,
        src_lang="en",
        tgt_lang="es",
        seed=42,
        show_progress=False,
    )

    summary = report.summary()
    assert "level1.recall@1" in summary.columns
    assert summary.loc["fake/model", "level1.recall@1"] == 1.0
    assert report.run_meta["dataset_hash"] == report.models[0].meta["dataset_hash"]
    assert report.models[0].selection is None

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    payload = json.loads(report.to_json(json_path))
    html = report.to_html(html_path)

    assert payload["models"][0]["selection"] is None
    assert payload["models"][0]["meta"]["embedprobe_version"]
    assert "sortTable" in html
    assert json_path.exists()
    assert html_path.exists()


def test_cli_model_normalization_accepts_comma_and_repeated_values():
    assert _normalize_models(["a,b", "c"]) == ["a", "b", "c"]
