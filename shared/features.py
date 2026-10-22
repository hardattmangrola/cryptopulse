"""
CryptoPulse - Shared Feature Engineering and Labeling Module
This module is shared between training and production inference to guarantee identical
feature definitions, calculations, preprocessing, and order.

Key Principles:
- Causality: Only uses information available at or before the current candle timestamp.
- Uniformity: Both training and inference use the same FEATURE_COLUMNS order.
- Clean Labeling: Generates 3-class movement target (UP, STABLE, DOWN) based on future 15m return.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# The exact list of engineered features used by the ML model.
# Order is strictly preserved between training and inference.
FEATURE_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1m",
    "return_5m",
    "return_15m",
    "sma_5",
    "sma_15",
    "sma_60",
    "ema_5",
    "ema_15",
    "rolling_std_15",
    "rsi_14",
    "volume_change_1m",
    "volume_ma_15",
    "hl_spread",
    "hour",
    "day_of_week",
]

# Minimum number of completed 1-minute candles required to compute all rolling features
# (Governed by sma_60, which requires 60 observations)
MIN_CANDLES_FOR_FEATURES: int = 60

# Default target labeling parameters (1-minute prediction horizon)
DEFAULT_PREDICTION_HORIZON_MINUTES: int = 1
DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.0005  # 0.05% return in 1 minute

# Target class representations
LABEL_DOWN: int = 0
LABEL_STABLE: int = 1
LABEL_UP: int = 2

LABEL_INT_TO_NAME: Dict[int, str] = {
    LABEL_DOWN: "DOWN",
    LABEL_STABLE: "STABLE",
    LABEL_UP: "UP",
}

LABEL_NAME_TO_INT: Dict[str, int] = {
    "DOWN": LABEL_DOWN,
    "STABLE": LABEL_STABLE,
    "UP": LABEL_UP,
}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI) for a price series.

    Args:
        series: Price series (e.g. Close prices).
        period: Rolling window period (default 14).

    Returns:
        pd.Series: RSI values bounded between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero by adding small epsilon
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Transforms raw OHLCV market candles into the standardized feature set.
    Safe for both training DataFrames and production streaming buffers.

    Args:
        df: DataFrame containing at least ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            (case-insensitive).
        drop_na: If True, drops initial rows with NaN caused by rolling calculations.
                 Set to False when inspecting single rows or managing buffer alignment.

    Returns:
        pd.DataFrame: Contains all FEATURE_COLUMNS in exact order, plus Timestamp.
    """
    # Standardize column names to lowercase
    col_map = {col: col.lower() for col in df.columns}
    work_df = df.rename(columns=col_map).copy()

    # Automatically map candle_timestamp to timestamp if present
    if "timestamp" not in work_df.columns and "candle_timestamp" in work_df.columns:
        work_df["timestamp"] = work_df["candle_timestamp"]

    # Validate essential input columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in work_df.columns:
            raise ValueError(f"Missing required raw column '{col}' for feature calculation.")

    # Sort strictly chronologically
    work_df = work_df.sort_values(by="timestamp").reset_index(drop=True)

    close = work_df["close"]
    high = work_df["high"]
    low = work_df["low"]
    volume = work_df["volume"]

    # 1. Price Returns (1m, 5m, 15m)
    work_df["return_1m"] = close.pct_change(1)
    work_df["return_5m"] = close.pct_change(5)
    work_df["return_15m"] = close.pct_change(15)

    # 2. Simple Moving Averages (SMA)
    work_df["sma_5"] = close.rolling(window=5, min_periods=5).mean()
    work_df["sma_15"] = close.rolling(window=15, min_periods=15).mean()
    work_df["sma_60"] = close.rolling(window=60, min_periods=60).mean()

    # 3. Exponential Moving Averages (EMA)
    work_df["ema_5"] = close.ewm(span=5, adjust=False).mean()
    work_df["ema_15"] = close.ewm(span=15, adjust=False).mean()

    # 4. Volatility / Rolling standard deviation
    work_df["rolling_std_15"] = close.rolling(window=15, min_periods=15).std()

    # 5. Relative Strength Index (RSI)
    work_df["rsi_14"] = compute_rsi(close, period=14)

    # 6. Volume indicators
    # When previous volume is 0.0, pct_change produces inf; replace with 0.0
    work_df["volume_change_1m"] = volume.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    work_df["volume_ma_15"] = volume.rolling(window=15, min_periods=15).mean()

    # 7. High-Low Price Spread
    work_df["hl_spread"] = (high - low) / (close + 1e-10)

    # 8. Calendar / Cyclical features derived from timestamp
    dt_series = pd.to_datetime(work_df["timestamp"], unit="s", utc=True)
    work_df["hour"] = dt_series.dt.hour
    work_df["day_of_week"] = dt_series.dt.dayofweek

    # Clean any accidental inf/-inf across all feature columns
    for col in FEATURE_COLUMNS:
        if col in work_df.columns:
            work_df[col] = work_df[col].replace([np.inf, -np.inf], np.nan)

    if drop_na:
        work_df = work_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # Reorder columns to guarantee exact feature order
    output_cols = ["timestamp"] + [col for col in FEATURE_COLUMNS if col != "timestamp"]
    return work_df[output_cols]


def generate_target_labels(
    df: pd.DataFrame,
    horizon_minutes: int = DEFAULT_PREDICTION_HORIZON_MINUTES,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> pd.DataFrame:
    """
    Generates 3-class target labels for supervised training based on future return:
        future_return = (future_close - current_close) / current_close
        UP: future_return > +threshold (e.g. > +0.2%) -> Label 2
        DOWN: future_return < -threshold (e.g. < -0.2%) -> Label 0
        STABLE: -threshold <= future_return <= +threshold -> Label 1

    Args:
        df: DataFrame containing at least 'timestamp' and 'close' (or 'Close').
        horizon_minutes: How many minutes into the future to measure outcome (default: 15).
        threshold: Return boundary for UP / DOWN classification (default: 0.002 = 0.2%).

    Returns:
        pd.DataFrame: Contains original columns plus 'future_close', 'future_return',
                      'target' (int), and 'target_name' (str).
                      Rows with unknown future outcomes (last horizon rows) are dropped.
    """
    col_map = {col: col.lower() for col in df.columns}
    work_df = df.rename(columns=col_map).copy()

    # Look ahead 'horizon_minutes' rows
    work_df["future_close"] = work_df["close"].shift(-horizon_minutes)
    work_df["future_return"] = (work_df["future_close"] - work_df["close"]) / work_df["close"]

    # Assign 3-class target
    # Default to STABLE (1)
    work_df["target"] = LABEL_STABLE
    # UP (2)
    work_df.loc[work_df["future_return"] > threshold, "target"] = LABEL_UP
    # DOWN (0)
    work_df.loc[work_df["future_return"] < -threshold, "target"] = LABEL_DOWN

    work_df["target_name"] = work_df["target"].map(LABEL_INT_TO_NAME)

    # Drop the trailing rows where future_close cannot be observed
    labelled_df = work_df.dropna(subset=["future_close", "future_return"]).reset_index(drop=True)
    return labelled_df


def prepare_training_features_and_targets(
    raw_df: pd.DataFrame,
    horizon_minutes: int = DEFAULT_PREDICTION_HORIZON_MINUTES,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    End-to-end pipeline to compute features and target labels from raw candles.

    Returns:
        Tuple of:
          - X (pd.DataFrame): Feature matrix with FEATURE_COLUMNS
          - y (pd.Series): Target integer labels (0: DOWN, 1: STABLE, 2: UP)
          - full_df (pd.DataFrame): Combined DataFrame including timestamps and returns
    """
    # 1. Compute features with drop_na=False first so target shifting aligns correctly
    feat_df = compute_features(raw_df, drop_na=False)

    # 2. Generate future targets
    labelled_df = generate_target_labels(feat_df, horizon_minutes=horizon_minutes, threshold=threshold)

    # 3. Drop any remaining NaNs (from initial rolling window)
    clean_df = labelled_df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)

    X = clean_df[FEATURE_COLUMNS].copy()
    y = clean_df["target"].astype(int).copy()

    return X, y, clean_df
