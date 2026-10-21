"""Unit tests for inference and prediction functions."""

import pytest
from newsclf.data import LABEL_NAMES
from newsclf.predict import predict, predict_batch


def test_predict_structure():
    """Verify predict returns dictionary with required keys and valid types."""
    res = predict("NASA reveals new images from deep space telescope")
    assert isinstance(res, dict)
    assert "label" in res
    assert "confidence" in res
    assert "top_k_probs" in res

    assert res["label"] in LABEL_NAMES
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0

    assert isinstance(res["top_k_probs"], dict)
    assert len(res["top_k_probs"]) == 4
    for label, prob in res["top_k_probs"].items():
        assert label in LABEL_NAMES
        assert 0.0 <= prob <= 1.0

    prob_sum = sum(res["top_k_probs"].values())
    assert 0.99 <= prob_sum <= 1.01


def test_predict_sports_acceptance():
    """Test acceptance criterion: 'Messi scores twice in Champions League final' returns Sports."""
    res = predict("Messi scores twice in Champions League final")
    assert res["label"] == "Sports"
    assert res["confidence"] > 0.80


def test_predict_business():
    """Test business headline prediction."""
    res = predict("Wall Street opens higher as Federal Reserve holds interest rates steady")
    assert res["label"] == "Business"
    assert res["confidence"] > 0.60


def test_predict_scitech():
    """Test Sci/Tech headline prediction."""
    res = predict("OpenAI releases new artificial intelligence neural network software update")
    assert res["label"] == "Sci/Tech"
    assert res["confidence"] > 0.60


def test_predict_world():
    """Test World headline prediction."""
    res = predict("Prime Minister announces diplomatic treaty summit in Geneva with foreign leaders")
    assert res["label"] == "World"
    assert res["confidence"] > 0.50


def test_predict_empty_input():
    """Test handling of empty string."""
    res = predict("")
    assert res["label"] == "Unknown"
    assert res["confidence"] == 0.0


def test_predict_batch():
    """Test batch prediction function."""
    headlines = [
        "Lakers win NBA championship after thrilling overtime victory",
        "Stock market shares plunge as tech giants miss earnings targets",
    ]
    results = predict_batch(headlines)
    assert len(results) == 2
    assert results[0]["label"] == "Sports"
    assert results[1]["label"] == "Business"


def test_explain_prediction_sports():
    """Test explain_prediction generates top positive feature contributions."""
    from newsclf.predict import explain_prediction

    exp = explain_prediction("Messi scores twice in Champions League final as team secures victory")
    assert isinstance(exp, list)
    assert len(exp) > 0
    features = [item["feature"] for item in exp]
    assert any(term in features for term in ["league", "champions", "victory", "team", "final"])
    for item in exp:
        assert "feature" in item
        assert "contribution" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["contribution"], float)
        assert item["contribution"] > 0


def test_explain_prediction_empty_and_stopwords():
    """Test explain_prediction with empty input and stopword filtering."""
    from newsclf.predict import explain_prediction

    assert explain_prediction("") == []
    assert explain_prediction(None) == []

    # Text containing stopwords: stopwords like 'in', 'the' should not be features
    exp = explain_prediction("Champions league in the final")
    features = [item["feature"] for item in exp]
    assert "in" not in features
    assert "the" not in features


def test_predict_with_explain_flag():
    """Test predict() with explain=True includes explanation in return dict."""
    res = predict("Federal Reserve lowers interest rates following quarterly earnings", explain=True)
    assert "explanation" in res
    assert isinstance(res["explanation"], list)
    assert len(res["explanation"]) > 0
    assert res["label"] == "Business"
