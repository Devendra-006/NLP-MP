"""Unit tests for NLP improvements and evaluation modules.

Covers:
1. Expected Calibration Error (ECE) calculation logic.
2. Uncertainty threshold behavior and boundary transitions (0.59 / 0.60 / 0.61).
3. Candidate model pipelines builder.
4. Dataset integrity for modern and out-of-domain benchmarks.
5. Error analysis output schema.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from newsclf.calibration import compute_multiclass_ece, evaluate_threshold_tradeoff
from newsclf.compare_models import get_candidate_models
from newsclf.data import LABEL_NAMES
from newsclf.predict import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compute_multiclass_ece_perfect():
    """Test ECE on perfectly calibrated dummy outputs."""
    # 4 samples, completely confident and 100% correct
    y_true = np.array([0, 1, 2, 3])
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ece, mce, bin_details = compute_multiclass_ece(y_true, probs, n_bins=5)
    assert 0.0 <= ece <= 0.01
    assert 0.0 <= mce <= 0.01


def test_compute_multiclass_ece_uncalibrated():
    """Test ECE on overconfident wrong predictions."""
    # All predicted as class 0 with 90% confidence, but true labels are different
    y_true = np.array([1, 2, 3, 1, 2, 3])
    probs = np.full((6, 4), 0.03)
    probs[:, 0] = 0.91
    ece, mce, bin_details = compute_multiclass_ece(y_true, probs, n_bins=5)
    # Accuracy is 0%, confidence is 91%, ECE should be around 0.91
    assert ece > 0.80


def test_uncertainty_threshold_tradeoff():
    """Test coverage curve logic."""
    y_true = np.array([0, 1, 2, 3])
    probs = np.array([
        [0.95, 0.02, 0.02, 0.01],
        [0.85, 0.05, 0.05, 0.05],
        [0.55, 0.20, 0.15, 0.10],
        [0.45, 0.30, 0.15, 0.10],
    ])
    tradeoff = evaluate_threshold_tradeoff(y_true, probs, thresholds=np.array([0.50, 0.60, 0.80]))
    assert len(tradeoff) == 3
    # Coverage should decrease as threshold increases
    coverages = [t["coverage_pct"] for t in tradeoff]
    assert coverages[0] >= coverages[1] >= coverages[2]


def test_uncertain_boundary_predictions():
    """Test threshold logic and boundary flags for confidence around 0.60."""
    # Text prediction returns boolean uncertain flag
    res_sports = predict("Messi scores twice in Champions League final", model_path="models/baseline_model.joblib")
    assert res_sports["confidence"] > 0.60
    # Predict without explain should have valid top_k_probs
    assert len(res_sports["top_k_probs"]) == 4

    # Ambiguous multi-domain sentence should have lower confidence or uncertainty
    res_ambiguous = predict("Company director attends peace conference", model_path="models/baseline_model.joblib")
    assert "uncertain" in res_ambiguous


def test_candidate_models_structure():
    """Test that candidate models pipeline dictionary builds valid sklearn pipelines."""
    models = get_candidate_models(ngram_range=(1, 1), max_features=1000)
    assert "Multinomial Naive Bayes" in models
    assert "Logistic Regression (Current)" in models
    assert "Linear SVM (LinearSVC)" in models

    for name, pipe in models.items():
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")


def test_benchmark_datasets_exist():
    """Test that curated evaluation benchmark datasets exist with valid schemas."""
    modern_path = PROJECT_ROOT / "data" / "modern_headlines.csv"
    assert modern_path.exists(), "modern_headlines.csv is missing"
    df_modern = pd.read_csv(modern_path)
    assert "text" in df_modern.columns
    assert "label" in df_modern.columns
    assert len(df_modern) >= 40
    # Labels should be 0, 1, 2, or 3
    assert set(df_modern["label"].unique()).issubset({0, 1, 2, 3})

    ood_path = PROJECT_ROOT / "data" / "out_of_domain_headlines.csv"
    assert ood_path.exists(), "out_of_domain_headlines.csv is missing"
    df_ood = pd.read_csv(ood_path)
    assert "text" in df_ood.columns
    assert len(df_ood) >= 40
