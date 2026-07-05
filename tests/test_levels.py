import numpy as np

from embedprobe.levels.level0 import signal_to_noise
from embedprobe.levels.level1 import retrieval_metrics, true_ranks
from embedprobe.levels.level2 import topic_structure
from embedprobe.levels.level3 import error_taxonomy
from embedprobe.taxonomy import classify_miss, jaccard


def test_level0_reports_signal_noise_and_verdict_fields():
    sim = np.array(
        [
            [0.95, 0.10, 0.20, 0.00],
            [0.05, 0.90, 0.15, 0.10],
            [0.20, 0.10, 0.92, 0.05],
            [0.00, 0.15, 0.10, 0.93],
        ]
    )

    result = signal_to_noise(sim, seed=7)

    metrics = result["metrics"]
    assert metrics["mean_true"] > metrics["mean_random"]
    assert metrics["ks_statistic"] >= 0
    assert metrics["p_value"] == metrics["ks_p_value"]
    assert 0 <= metrics["overlap_fraction"] <= 1
    assert metrics["verdict"] in {"strong signal", "weak signal", "poor separation"}


def test_level1_computes_exact_ranks_recall_mrr_and_histogram():
    sim = np.array(
        [
            [0.90, 0.10, 0.20],
            [0.80, 0.70, 0.10],
            [0.20, 0.10, 0.95],
        ]
    )

    result = retrieval_metrics(sim, ks=(1, 2))

    assert true_ranks(sim).tolist() == [1, 2, 1]
    assert result["metrics"]["recall@1"] == 2 / 3
    assert result["metrics"]["recall@2"] == 1.0
    assert result["metrics"]["precision@2"] == 0.5
    assert np.isclose(result["metrics"]["mrr"], (1 + 0.5 + 1) / 3)
    assert "similarity_histogram" in result["data"]


def test_level2_reports_topic_confusion_and_cohesion():
    sim = np.array(
        [
            [0.90, 0.10, 0.20, 0.00],
            [0.80, 0.70, 0.10, 0.00],
            [0.10, 0.00, 0.95, 0.30],
            [0.05, 0.00, 0.20, 0.90],
        ]
    )
    topics = ["legal", "legal", "medical", "medical"]

    result = topic_structure(sim, topics)

    assert result["data"]["topics"] == ["legal", "medical"]
    assert result["metrics"]["topic_retrieval_accuracy"] == 1.0
    assert result["metrics"]["topic_cohesion"] > result["metrics"]["topic_leakage"]


def test_level3_classifies_misses_with_jaccard_and_levenshtein_records():
    sim = np.array(
        [
            [0.90, 0.10, 0.20],
            [0.80, 0.70, 0.10],
            [0.20, 0.10, 0.95],
        ]
    )
    src = ["q0", "q1", "q2"]
    tgt = ["red car fast", "red car", "blue ocean"]
    topics = ["transport", "transport", "nature"]

    result = error_taxonomy(sim, src, tgt, topics)

    assert result["metrics"]["n_misses"] == 1
    miss = result["data"]["misses"][0]
    assert miss["query_index"] == 1
    assert miss["error_type"] == "lexical_confusion"
    assert miss["gold_rank"] == miss["true_rank"] == 2
    assert np.isclose(miss["jaccard"], 2 / 3)
    assert 0 <= miss["levenshtein"] <= 1
    assert miss["query_topic"] == miss["retrieved_topic"] == "transport"


def test_taxonomy_thresholds_are_parameterized():
    score, error_type = classify_miss("alpha beta", "alpha gamma", high_thresh=0.3, low_thresh=0.2)

    assert np.isclose(score, jaccard("alpha beta", "alpha gamma"))
    assert error_type == "lexical_confusion"
