"""
Unit tests for 3-class target movement labeling in shared/features.py.
Verifies future return calculation, UP/DOWN/STABLE thresholds, and horizon alignment.
"""

import pytest
import pandas as pd
import numpy as np
from shared.features import (
    generate_target_labels,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
)


def test_target_labeling_logic():
    """Verify correct class assignment based on configurable threshold."""
    # Construct sequence where 15 steps ahead has known return:
    # Row 0: price = 100
    # Horizon = 2 (for concise unit test)
    # Row 2 (horizon for row 0): price = 101.0 -> return = +1.0% (> +0.2% -> UP)
    # Row 3 (horizon for row 1): price = 98.0 -> return = -2.0% (< -0.2% -> DOWN)
    # Row 4 (horizon for row 2): price = 101.1 -> return = +0.099% (between -0.2% and +0.2% -> STABLE)
    df = pd.DataFrame({
        "timestamp": [10, 20, 30, 40, 50],
        "close": [100.0, 100.0, 101.0, 98.0, 101.1],
    })

    labelled = generate_target_labels(df, horizon_minutes=2, threshold=0.002)

    # We have 5 rows and horizon is 2, so the first 3 rows receive labels, and last 2 rows are dropped
    assert len(labelled) == 3

    # Row 0 -> future is index 2 (101.0) -> +1% -> UP (2)
    assert labelled.loc[0, "target"] == LABEL_UP
    assert labelled.loc[0, "target_name"] == "UP"

    # Row 1 -> future is index 3 (98.0) -> -2% -> DOWN (0)
    assert labelled.loc[1, "target"] == LABEL_DOWN
    assert labelled.loc[1, "target_name"] == "DOWN"

    # Row 2 -> future is index 4 (101.1) -> +0.099% -> STABLE (1)
    assert labelled.loc[2, "target"] == LABEL_STABLE
    assert labelled.loc[2, "target_name"] == "STABLE"


def test_trailing_horizon_rows_are_dropped():
    """Verify that rows without observable future prices are excluded from labelled set."""
    df = pd.DataFrame({
        "timestamp": list(range(20)),
        "close": [50000.0] * 20,
    })
    labelled = generate_target_labels(df, horizon_minutes=15)
    # 20 rows - 15 horizon rows = 5 rows can be evaluated
    assert len(labelled) == 5
    assert not labelled["future_close"].isnull().any()
