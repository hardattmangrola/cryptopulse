"""
Unit and integration tests for Phase 7 Continuous Maintenance:
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift calculation
- Delayed 1-minute ground-truth evaluation
- Retraining controller with cooldown logic
- Monitoring microservice FastAPI endpoints
"""

import time
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from shared.database import Base, MarketData, Prediction, PredictionOutcome
from shared.features import FEATURE_COLUMNS, compute_features
from services.monitoring.drift import calculate_psi, analyze_feature_drift
from services.monitoring.evaluator import (
    classify_return,
    evaluate_pending_predictions,
    calculate_rolling_performance,
)
from services.monitoring.controller import RetrainingController
from services.monitoring.app import app


@pytest.fixture
def in_memory_db():
    """Sets up an isolated in-memory SQLite database for test runs."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


# -------------------------------------------------------------
# 1. Population Stability Index (PSI) & Drift Tests
# -------------------------------------------------------------
def test_psi_identical_distribution_is_near_zero():
    """Identical distributions should have PSI well below the NO_DRIFT threshold (< 0.10)."""
    bin_edges = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    # Sample exactly 100 points in each of the 10 bins
    values = np.concatenate([np.linspace(i * 10 + 1, i * 10 + 9, 100) for i in range(10)])
    psi = calculate_psi(values, bin_edges)
    assert psi < 0.05, f"Expected PSI < 0.05 for uniform distribution, got {psi}"
    assert psi < 0.10, "Should be categorized as NO_DRIFT (< 0.10)"


def test_psi_shifted_distribution_triggers_drift():
    """Severely shifted distribution should exceed PSI threshold 0.25."""
    bin_edges = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    # Sample all values in the highest bin (>90)
    values = np.array([95.0] * 500)
    psi = calculate_psi(values, bin_edges)
    assert psi >= 0.25, f"Expected PSI >= 0.25 for shifted distribution, got {psi}"


def test_analyze_feature_drift_with_mock_reference():
    """Tests feature drift detection against a mock reference distribution."""
    mock_ref = {
        "rsi_14": {
            "mean": 50.0,
            "std": 10.0,
            "quantiles": {
                "0.1": 35.0, "0.2": 40.0, "0.3": 45.0, "0.4": 48.0,
                "0.5": 50.0, "0.6": 52.0, "0.7": 55.0, "0.8": 60.0, "0.9": 65.0,
            }
        }
    }
    # Create production DataFrame where rsi_14 is heavily drifted (values around 90)
    prod_df = pd.DataFrame({
        "rsi_14": np.random.normal(90, 5, 200)
    })

    result = analyze_feature_drift(prod_df, ref_distribution=mock_ref)
    assert result["drift_detected"] is True
    assert "rsi_14" in result["drifted_features"]
    assert result["feature_metrics"]["rsi_14"]["drift_status"] == "SIGNIFICANT_DRIFT"


# -------------------------------------------------------------
# 2. Delayed 1-Minute Ground Truth Evaluation Tests
# -------------------------------------------------------------
def test_classify_return():
    """Verifies 3-class return classification threshold (0.05% return in 1m)."""
    assert classify_return(0.001) == "UP"
    assert classify_return(-0.001) == "DOWN"
    assert classify_return(0.0001) == "STABLE"
    assert classify_return(-0.0001) == "STABLE"


def test_delayed_evaluation_workflow(in_memory_db):
    """
    Simulates making a prediction at t0=1000 and evaluating with completed candle at t1=1060.
    """
    db = in_memory_db
    t0 = 1000
    t1 = 1060  # exactly 60s later

    # 1. Insert historical candle at t0 and future candle at t1
    c0 = MarketData(symbol="BTCUSDT", candle_timestamp=t0, open=70000, high=70050, low=69950, close=70000, volume=10)
    # Price rises from 70,000 to 70,100 (+0.14% return -> UP)
    c1 = MarketData(symbol="BTCUSDT", candle_timestamp=t1, open=70000, high=70150, low=69990, close=70100, volume=15)
    db.add_all([c0, c1])
    db.commit()

    # 2. Insert pending prediction made at t0
    pred = Prediction(
        prediction_timestamp=t0,
        symbol="BTCUSDT",
        predicted_class="UP",
        class_probabilities={"DOWN": 0.1, "STABLE": 0.2, "UP": 0.7},
        model_version="1.0.0",
        evaluation_status="PENDING",
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    # 3. Run evaluation (pretending current time is t1 + 10s)
    summary = evaluate_pending_predictions(db, horizon_seconds=60, threshold=0.0005)

    assert summary["evaluated_count"] == 1
    assert summary["correct_count"] == 1
    assert summary["batch_accuracy"] == 1.0

    # 4. Check that prediction is now marked EVALUATED
    db.refresh(pred)
    assert pred.evaluation_status == "EVALUATED"

    # 5. Check outcome table record
    outcome = db.query(PredictionOutcome).filter_by(prediction_id=pred.id).first()
    assert outcome is not None
    assert outcome.actual_class == "UP"
    assert outcome.is_correct is True
    assert outcome.actual_return > 0.001


def test_rolling_performance_metrics(in_memory_db):
    """Tests rolling Macro F1 and accuracy computation over ground-truth outcomes."""
    db = in_memory_db

    # Add 4 predictions and outcomes
    classes = ["UP", "DOWN", "STABLE", "UP"]
    actuals = ["UP", "DOWN", "DOWN", "UP"]  # 3 correct, 1 incorrect

    for i in range(4):
        pred = Prediction(
            prediction_timestamp=1000 + i * 60,
            symbol="BTCUSDT",
            predicted_class=classes[i],
            class_probabilities={"DOWN": 0.33, "STABLE": 0.33, "UP": 0.34},
            model_version="1.0.0",
            evaluation_status="EVALUATED",
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)

        outcome = PredictionOutcome(
            prediction_id=pred.id,
            evaluation_timestamp=1000 + i * 60 + 60,
            future_close=70000.0,
            actual_return=0.001 if actuals[i] == "UP" else -0.001,
            actual_class=actuals[i],
            is_correct=(classes[i] == actuals[i]),
        )
        db.add(outcome)
        db.commit()

    rolling = calculate_rolling_performance(db, window=10)
    assert rolling["sample_size"] == 4
    assert rolling["rolling_accuracy"] == 0.75  # 3/4
    assert rolling["rolling_macro_f1"] is not None


# -------------------------------------------------------------
# 3. Retraining Controller & Cooldown Tests
# -------------------------------------------------------------
def test_retraining_controller_cooldown():
    """Controller must enforce cooldown window between retraining runs."""
    ctrl = RetrainingController(cooldown_minutes=10)
    in_cd, _ = ctrl.is_in_cooldown()
    assert in_cd is False

    # Simulate triggering retraining
    ctrl.last_retrain_time = time.time()
    in_cd, rem = ctrl.is_in_cooldown()
    assert in_cd is True
    assert rem > 500

    # Attempting to trigger during cooldown should be blocked
    res = ctrl.trigger_retraining(reason="test_drift", force=False)
    assert res["status"] == "blocked_by_cooldown"
    assert res["seconds_remaining"] > 0


def test_retraining_trigger_conditions():
    """Tests evaluating whether drift or degraded performance triggers retraining."""
    ctrl = RetrainingController(min_macro_f1=0.30)

    # Condition 1: Significant drift
    drift_data = {"drift_detected": True, "max_psi": 0.32, "max_drift_feature": "rsi_14"}
    should_trig, reason = ctrl.evaluate_retraining_trigger(drift_results=drift_data)
    assert should_trig is True
    assert "drift" in reason

    # Condition 2: Performance degradation
    degraded_metrics = {"sample_size": 20, "rolling_macro_f1": 0.22}
    should_trig, reason = ctrl.evaluate_retraining_trigger(rolling_metrics=degraded_metrics)
    assert should_trig is True
    assert "performance_degradation" in reason

    # Condition 3: Normal conditions -> no trigger
    normal_metrics = {"sample_size": 20, "rolling_macro_f1": 0.45}
    should_trig, _ = ctrl.evaluate_retraining_trigger(rolling_metrics=normal_metrics)
    assert should_trig is False


# -------------------------------------------------------------
# 4. Monitoring Microservice REST API Tests
# -------------------------------------------------------------
def test_monitoring_microservice_health_and_metrics():
    """Tests FastAPI /health and /metrics endpoints on monitoring service."""
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["service"] == "monitoring"

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "cryptopulse_" in metrics_resp.text
