"""
Unit tests for shared/database.py.
Tests SQLAlchemy ORM models, table initialization, duplicate candle prevention,
prediction and outcome persistence, model run metadata, and drift metrics storage.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shared.database import (
    Base,
    MarketData,
    Prediction,
    PredictionOutcome,
    ModelRun,
    DriftMetric,
    save_market_candles,
    record_prediction,
    record_prediction_outcome,
    record_model_run,
    record_drift_metrics,
)


@pytest.fixture
def db_session():
    """Provides a fresh, isolated in-memory SQLite database session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_save_market_candles_and_deduplication(db_session):
    """Verify that candles are stored and duplicate timestamps for same symbol are discarded."""
    candles = [
        {"symbol": "BTCUSDT", "timestamp": 1000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 1.2, "source": "binance_rest"},
        {"symbol": "BTCUSDT", "timestamp": 1060, "open": 50050.0, "high": 50200.0, "low": 50000.0, "close": 50150.0, "volume": 2.1, "source": "binance_rest"},
    ]

    inserted_first = save_market_candles(db_session, candles)
    assert inserted_first == 2

    # Attempt to insert identical candles again (simulating overlap in polling)
    inserted_second = save_market_candles(db_session, candles)
    assert inserted_second == 0

    # Total records in table must remain 2
    count = db_session.query(MarketData).count()
    assert count == 2


def test_record_prediction_and_delayed_outcome(db_session):
    """Verify storing an inference prediction and its subsequent delayed ground-truth evaluation."""
    # 1. Store prediction
    pred_data = {
        "prediction_timestamp": 1700000000,
        "symbol": "BTCUSDT",
        "predicted_class": "UP",
        "class_probabilities": {"DOWN": 0.10, "STABLE": 0.25, "UP": 0.65},
        "model_version": "1.0.0",
    }
    pred_record = record_prediction(db_session, pred_data)

    assert pred_record.id is not None
    assert pred_record.evaluation_status == "PENDING"
    assert pred_record.predicted_class == "UP"

    # 2. Store delayed outcome (15 minutes later)
    outcome_data = {
        "prediction_id": pred_record.id,
        "evaluation_timestamp": 1700000900,
        "future_close": 50500.0,
        "actual_return": 0.005,
        "actual_class": "UP",
        "is_correct": True,
    }
    outcome_record = record_prediction_outcome(db_session, outcome_data)

    assert outcome_record.id is not None
    assert outcome_record.is_correct is True

    # 3. Verify prediction status has changed to EVALUATED
    updated_pred = db_session.query(Prediction).filter_by(id=pred_record.id).first()
    assert updated_pred.evaluation_status == "EVALUATED"
    assert updated_pred.outcome.actual_class == "UP"


def test_record_model_run(db_session):
    """Verify storing model training metadata, macro F1, and promotion status."""
    run_data = {
        "model_version": "1.0.0",
        "dataset_description": "Historical 1-minute BTC (last 100k rows)",
        "accuracy": 0.853,
        "macro_f1": 0.342,
        "training_status": "COMPLETED",
        "promotion_status": "PROMOTED",
        "mlflow_run_id": "test_run_12345",
        "parameters": {"n_estimators": 100, "max_depth": 5},
        "metrics": {"test_accuracy": 0.853, "test_macro_f1": 0.342},
    }
    run = record_model_run(db_session, run_data)

    assert run.id is not None
    assert run.model_version == "1.0.0"
    assert run.macro_f1 == 0.342
    assert run.promotion_status == "PROMOTED"

    # Test update existing version
    run_data["macro_f1"] = 0.355
    updated_run = record_model_run(db_session, run_data)
    assert updated_run.id == run.id
    assert updated_run.macro_f1 == 0.355


def test_record_drift_metrics(db_session):
    """Verify storing feature drift statistics and statuses."""
    drift_records = [
        {"timestamp": 1700001000, "feature_name": "rsi_14", "drift_statistic": 0.045, "drift_status": "NO_DRIFT"},
        {"timestamp": 1700001000, "feature_name": "close", "drift_statistic": 0.210, "drift_status": "DRIFT_DETECTED"},
    ]
    count = record_drift_metrics(db_session, drift_records)
    assert count == 2

    records = db_session.query(DriftMetric).all()
    assert len(records) == 2
    assert records[1].drift_status == "DRIFT_DETECTED"
