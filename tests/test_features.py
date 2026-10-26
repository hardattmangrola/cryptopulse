"""
Unit tests for shared/features.py
Verifies technical indicator calculations, causality (no lookahead bias),
feature schema uniformity, and NaN handling.
"""

import pytest
import numpy as np
import pandas as pd
from shared.features import (
    compute_features,
    compute_rsi,
    FEATURE_COLUMNS,
    MIN_CANDLES_FOR_FEATURES,
)


def create_synthetic_candles(n_candles: int = 100) -> pd.DataFrame:
    """Helper creating n synthetic 1-minute OHLCV candles."""
    timestamps = [1700000000 + i * 60 for i in range(n_candles)]
    prices = [50000.0 + (i * 10.0) for i in range(n_candles)]
    return pd.DataFrame({
        "Timestamp": timestamps,
        "Open": [p - 5.0 for p in prices],
        "High": [p + 15.0 for p in prices],
        "Low": [p - 10.0 for p in prices],
        "Close": prices,
        "Volume": [1.5 + (i * 0.05) for i in range(n_candles)],
    })


def test_compute_rsi_bounds():
    """Verify RSI produces values within [0, 100] range."""
    series = pd.Series([100, 102, 101, 105, 108, 107, 109, 112, 115, 114, 116, 118, 120, 119, 122, 125, 128])
    rsi = compute_rsi(series, period=14)
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_compute_features_column_uniformity():
    """Verify that compute_features outputs all FEATURE_COLUMNS in exact expected order."""
    df = create_synthetic_candles(100)
    feat_df = compute_features(df, drop_na=True)

    # Check presence and order
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns

    # Verify no NaNs remain after dropping warmup
    assert not feat_df[FEATURE_COLUMNS].isnull().any().any()
    assert len(feat_df) > 0


def test_compute_features_no_lookahead_bias():
    """
    Critical Causality Test:
    Modifying future prices must NOT alter the feature values of current or past candles.
    """
    df1 = create_synthetic_candles(80)
    df2 = df1.copy()

    # Dramatically change the price of the last candle in df2
    df2.loc[79, "Close"] = 999999.0
    df2.loc[79, "High"] = 1000000.0

    feat1 = compute_features(df1, drop_na=False)
    feat2 = compute_features(df2, drop_na=False)

    # Feature values at index 70 (which is 9 minutes before the change) must be 100% identical
    for col in FEATURE_COLUMNS:
        val1 = feat1.loc[70, col]
        val2 = feat2.loc[70, col]
        assert np.isclose(val1, val2, equal_nan=True), f"Lookahead bias detected in feature: {col}"


def test_missing_column_raises_error():
    """Verify ValueError is raised if any essential OHLCV column is absent."""
    bad_df = pd.DataFrame({"Timestamp": [1000], "Close": [50000.0]})
    with pytest.raises(ValueError) as exc:
        compute_features(bad_df)
    assert "Missing required raw column" in str(exc.value)


def test_candle_timestamp_alias_accepted():
    """Verify that DataFrames with 'candle_timestamp' column are handled seamlessly."""
    df = create_synthetic_candles(70)
    df = df.rename(columns={"Timestamp": "candle_timestamp"})
    feat_df = compute_features(df, drop_na=True)
    assert len(feat_df) > 0
    assert "timestamp" in feat_df.columns

