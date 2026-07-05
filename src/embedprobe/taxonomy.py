"""Jaccard-based retrieval-miss error taxonomy.

Each retrieval miss is classified by the token overlap between the *retrieved*
target sentence and the *true* target sentence:

- overlap >= ``high_thresh``  -> ``lexical_confusion`` (fooled by surface overlap)
- overlap <= ``low_thresh``   -> ``semantic_confusion`` (drifted semantically)
- otherwise                   -> ``topic_boundary``
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

LEXICAL = "lexical_confusion"
SEMANTIC = "semantic_confusion"
TOPIC_BOUNDARY = "topic_boundary"
ERROR_TYPES = (LEXICAL, SEMANTIC, TOPIC_BOUNDARY)

DEFAULT_HIGH_THRESH = 0.30
DEFAULT_LOW_THRESH = 0.20

# Any CJK character means the text has no whitespace word boundaries.
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> List[str]:
    """Language-aware tokens: characters for CJK text, lowercased words otherwise."""
    text = str(text)
    if _CJK_RE.search(text):
        return list(text)
    return text.lower().split()


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two sentences."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized edit-similarity ratio via difflib.SequenceMatcher."""
    return SequenceMatcher(None, str(a), str(b)).ratio()


def overlap_score(true_text: str, retrieved_text: str, method: str = "jaccard") -> float:
    if method == "jaccard":
        return jaccard(true_text, retrieved_text)
    if method == "levenshtein":
        return levenshtein_ratio(true_text, retrieved_text)
    raise ValueError(f"Unknown overlap method: {method!r}")


def classify_miss(
    true_text: str,
    retrieved_text: str,
    high_thresh: float = DEFAULT_HIGH_THRESH,
    low_thresh: float = DEFAULT_LOW_THRESH,
    method: str = "jaccard",
) -> tuple:
    """Return ``(overlap_score, error_type)`` for one retrieval miss."""
    if low_thresh > high_thresh:
        raise ValueError("low_thresh must not exceed high_thresh")
    score = overlap_score(true_text, retrieved_text, method)
    if score >= high_thresh:
        error_type = LEXICAL
    elif score <= low_thresh:
        error_type = SEMANTIC
    else:
        error_type = TOPIC_BOUNDARY
    return score, error_type
