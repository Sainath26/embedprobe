import pandas as pd
import pytest

from embedprobe.data import dataset_hash, has_topics, load_pairs


def test_load_pairs_validates_required_columns_and_drops_empty_texts():
    df = pd.DataFrame(
        {
            "en": ["hello", "", None, "goodbye"],
            "es": ["hola", "vacio", "nulo", "adios"],
            "topic": ["greeting", "bad", "bad", "farewell"],
            "ignored": [1, 2, 3, 4],
        }
    )

    loaded = load_pairs(df, src_col="en", tgt_col="es", topic_col="topic")

    assert list(loaded.columns) == ["en", "es", "topic"]
    assert loaded["en"].tolist() == ["hello", "goodbye"]
    assert has_topics(loaded)


def test_load_pairs_raises_for_missing_required_column():
    df = pd.DataFrame({"en": ["hello"]})

    with pytest.raises(ValueError, match="missing required"):
        load_pairs(df, src_col="en", tgt_col="es")


def test_dataset_hash_is_stable_for_loaded_pairs():
    df = pd.DataFrame({"en": ["a", "b"], "es": ["x", "y"], "topic": ["t1", "t2"]})
    loaded = load_pairs(df, src_col="en", tgt_col="es", topic_col="topic")

    assert dataset_hash(loaded) == dataset_hash(loaded.copy())
    assert len(dataset_hash(loaded)) == 64
