"""Result containers: one ModelDiagnostics per model, bundled in a ProbeReport."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

LEVEL_KEYS = ("level0", "level1", "level2", "level3")


@dataclass
class ModelDiagnostics:
    """All four diagnostic levels for a single model."""

    model_name: str
    meta: Dict = field(default_factory=dict)
    level0: Optional[Dict] = None
    level1: Optional[Dict] = None
    level2: Optional[Dict] = None
    level3: Optional[Dict] = None
    selection: Optional[Dict] = None

    def metrics(self) -> Dict[str, float]:
        """Flat ``{level.metric: value}`` mapping across all computed levels."""
        flat: Dict[str, float] = {}
        for key in LEVEL_KEYS:
            level = getattr(self, key)
            if level:
                for name, value in level["metrics"].items():
                    flat[f"{key}.{name}"] = value
        return flat

    def to_dict(self) -> Dict:
        out = {"model": self.model_name, "meta": self.meta}
        for key in LEVEL_KEYS:
            level = getattr(self, key)
            if level:
                out[key] = level
        out["selection"] = self.selection
        return out


@dataclass
class ProbeReport:
    """Diagnostics for every probed model plus run-level metadata."""

    models: List[ModelDiagnostics]
    run_meta: Dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Cross-model metric table (models as rows, level.metric as columns)."""
        rows = {d.model_name: d.metrics() for d in self.models}
        return pd.DataFrame.from_dict(rows, orient="index")

    def to_dict(self) -> Dict:
        return {"run": self.run_meta, "models": [d.to_dict() for d in self.models]}

    def to_json(self, path: Union[str, Path, None] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_html(self, path: Union[str, Path, None] = None) -> str:
        from embedprobe.report.html import render_html

        html = render_html(self)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html
