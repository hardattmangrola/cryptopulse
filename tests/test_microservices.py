"""
Unit tests for Phase 4 Microservices (Ingestion, Inference, and Training).
Tests FastAPI endpoints, health checks, Prometheus metrics, prediction outputs,
model reloading, and quality gates using FastAPI TestClient with mocked dependencies.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from shared.database import Base, get_db, ModelRun, MarketData
from services.ingestion.app import app as ingestion_app
from services.inference.app import app as inference_app
from services.training.app import app as training_app


@pytest.fixture
def test_db_session():
    """Provides a fresh isolated SQLite in-memory database using StaticPool for thread-safety."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# =====================================================================
# Ingestion Microservice Tests
# =====================================================================

def test_ingestion_health_endpoint():
    """Verify GET /health on Ingestion microservice."""
    client = TestClient(ingestion_app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ingestion"
    assert "status" in data
    assert "poll_interval_seconds" in data


def test_ingestion_metrics_endpoint():
    """Verify GET /metrics on Ingestion microservice returns Prometheus format."""
    client = TestClient(ingestion_app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cryptopulse_http_requests_total" in response.text


def test_ingestion_manual_poll(test_db_session):
    """Verify POST /poll triggers data fetch and saves candles into database."""
    ingestion_app.dependency_overrides[get_db] = lambda: test_db_session
    client = TestClient(ingestion_app)

    mock_candle = {
        "timestamp": 1700000000,
        "open": 60000.0,
        "high": 60100.0,
        "low": 59900.0,
        "close": 60050.0,
        "volume": 5.0,
        "symbol": "BTCUSDT",
        "is_completed": True,
        "source": "binance_rest"
    }

    with patch("services.ingestion.app.binance_client.fetch_completed_candles", return_value=[mock_candle]):
        response = client.post("/poll")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["candles_fetched"] == 1
        assert data["new_candles_saved"] == 1


# =====================================================================
# Inference Microservice Tests
# =====================================================================

def test_inference_health_endpoint():
    """Verify GET /health on Inference microservice."""
    client = TestClient(inference_app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "inference"
    assert "model_loaded" in data


def test_inference_metrics_endpoint():
    """Verify GET /metrics on Inference microservice."""
    client = TestClient(inference_app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cryptopulse_active_model_version" in response.text


def test_inference_predict_insufficient_candles(test_db_session):
    """Verify POST /predict rejects requests with fewer than 60 completed candles."""
    inference_app.dependency_overrides[get_db] = lambda: test_db_session
    client = TestClient(inference_app)

    few_candles = [{"timestamp": 1000 + i * 60, "open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 1.0} for i in range(10)]
    response = client.post("/predict", json={"candles": few_candles})

    assert response.status_code == 400
    assert "Insufficient completed candles" in response.json()["detail"]


def test_inference_predict_success(test_db_session):
    """Verify POST /predict returns standard 3-class prediction and probabilities."""
    inference_app.dependency_overrides[get_db] = lambda: test_db_session
    client = TestClient(inference_app)

    # Generate 70 valid candles (>= 60)
    valid_candles = [
        {
            "timestamp": 1700000000 + i * 60,
            "open": 50000.0 + i,
            "high": 50010.0 + i,
            "low": 49990.0 + i,
            "close": 50005.0 + i,
            "volume": 2.5,
        }
        for i in range(70)
    ]

    response = client.post("/predict", json={"candles": valid_candles})
    assert response.status_code == 200
    data = response.json()

    assert data["symbol"] == "BTCUSDT"
    assert data["prediction"] in ["UP", "STABLE", "DOWN"]
    assert "DOWN" in data["probabilities"]
    assert "STABLE" in data["probabilities"]
    assert "UP" in data["probabilities"]

    # Verify probabilities sum to ~1.0
    probs = data["probabilities"]
    assert pytest.approx(probs["DOWN"] + probs["STABLE"] + probs["UP"], 0.01) == 1.0


def test_inference_predict_from_database(test_db_session):
    """Verify POST /predict succeeds when querying candles directly from PostgreSQL MarketData."""
    inference_app.dependency_overrides[get_db] = lambda: test_db_session
    client = TestClient(inference_app)

    # Insert 75 candles into MarketData
    for i in range(75):
        candle = MarketData(
            symbol="BTCUSDT",
            candle_timestamp=1700000000 + i * 60,
            open=50000.0 + i,
            high=50010.0 + i,
            low=49990.0 + i,
            close=50005.0 + i,
            volume=2.5,
        )
        test_db_session.add(candle)
    test_db_session.commit()

    # Call POST /predict with no body (reads from DB)
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["UP", "STABLE", "DOWN"]



def test_inference_reload_model():
    """Verify POST /reload-model reloads model weights from disk."""
    client = TestClient(inference_app)
    response = client.post("/reload-model")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_inference_web_ui_endpoint():
    """Verify GET / serves the interactive prediction Web UI HTML page."""
    client = TestClient(inference_app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CryptoPulse Real-Time Predictor" in response.text


# =====================================================================
# Training Microservice Tests
# =====================================================================

def test_training_health_endpoint():
    """Verify GET /health on Training microservice."""
    client = TestClient(training_app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "training"
    assert "status" in data


def test_training_quality_gate_decision(test_db_session):
    """Verify candidate model quality gate promotes when Macro F1 >= threshold."""
    training_app.dependency_overrides[get_db] = lambda: test_db_session
    client = TestClient(training_app)

    # Pre-insert initial model run record into test database
    initial_run = ModelRun(model_version="2.0.0-test", promotion_status="CANDIDATE")
    test_db_session.add(initial_run)
    test_db_session.commit()

    mock_metrics = {
        "val_macro_f1": 0.38,
        "val_accuracy": 0.80,
        "test_macro_f1": 0.36,
        "test_accuracy": 0.82,
        "test_macro_precision": 0.40,
        "test_macro_recall": 0.38,
    }

    mock_train_result = {
        "model_path": "models/baseline_model.joblib",
        "metrics": mock_metrics,
        "metadata": {"model_version": "2.0.0-test", "dataset_rows_used": 1000},
    }

    with patch("services.training.app.train_model", return_value=mock_train_result):
        # Case 1: Pass quality gate (0.36 >= 0.30)
        resp_pass = client.post("/train", json={"quality_gate_f1_threshold": 0.30, "sample_rows": 1000})
        assert resp_pass.status_code == 200
        data_pass = resp_pass.json()
        assert data_pass["quality_gate"]["passed"] is True
        assert data_pass["quality_gate"]["promotion_status"] == "PROMOTED"

        # Case 2: Reject quality gate (0.36 < 0.50)
        resp_reject = client.post("/train", json={"quality_gate_f1_threshold": 0.50, "sample_rows": 1000})
        assert resp_reject.status_code == 200
        data_reject = resp_reject.json()
        assert data_reject["quality_gate"]["passed"] is False
        assert data_reject["quality_gate"]["promotion_status"] == "REJECTED"
