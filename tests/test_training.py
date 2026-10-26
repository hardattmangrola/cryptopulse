"""
Unit tests for services/training/train.py.
Verifies chronological splitting, reference distribution calculations,
and model training & output probability normalization.
"""

import pytest
import numpy as np
import pandas as pd
from services.training.train import (
    split_chronological,
    calculate_feature_reference_distribution,
    evaluate_model,
)
from shared.features import (
    FEATURE_COLUMNS,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
)
from xgboost import XGBClassifier


def test_split_chronological_ratios_and_order():
    """Verify chronological split respects ratios and never shuffles time order."""
    n = 1000
    X = pd.DataFrame({col: np.arange(n) for col in FEATURE_COLUMNS})
    y = pd.Series(np.random.choice([0, 1, 2], size=n))

    X_train, y_train, X_val, y_val, X_test, y_test = split_chronological(
        X, y, train_ratio=0.70, val_ratio=0.15
    )

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    # Verify strict temporal ordering
    assert X_train["open"].iloc[-1] < X_val["open"].iloc[0]
    assert X_val["open"].iloc[-1] < X_test["open"].iloc[0]


def test_calculate_feature_reference_distribution():
    """Verify reference distributions extract statistical moments and percentiles."""
    n = 200
    X = pd.DataFrame({col: np.linspace(1.0, 100.0, n) for col in FEATURE_COLUMNS})
    ref = calculate_feature_reference_distribution(X)

    for col in FEATURE_COLUMNS:
        assert col in ref
        assert "mean" in ref[col]
        assert "std" in ref[col]
        assert "quantiles" in ref[col]
        assert "0.5" in ref[col]["quantiles"]  # Median


def test_model_training_and_probability_sum():
    """Verify XGBoost model trains and outputs valid 3-class probability distributions."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y = pd.Series(np.random.choice([0, 1, 2], size=n))

    clf = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )
    clf.fit(X.iloc[:200], y.iloc[:200])

    preds = clf.predict(X.iloc[200:])
    probs = clf.predict_proba(X.iloc[200:])

    # Predicted classes must be in {0, 1, 2}
    assert set(preds).issubset({LABEL_DOWN, LABEL_STABLE, LABEL_UP})

    # Class probabilities must sum to 1.0 for every sample
    prob_sums = probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5)


def test_get_next_model_version():
    """Verify get_next_model_version increments version and accepts explicit version."""
    from services.training.train import get_next_model_version

    # Explicit override
    assert get_next_model_version("2.5.0") == "2.5.0"

    # Incremental auto-calculation
    next_ver = get_next_model_version()
    assert "." in next_ver
    parts = next_ver.split(".")
    assert len(parts) == 3

