"""
CryptoPulse - Inference Microservice
Exposes REST endpoints via FastAPI to classify short-term Bitcoin price movement direction (UP, STABLE, DOWN).

Workflow:
1. Loads approved model artifact and metadata from disk (supports dynamic reload without restart).
2. On POST /predict:
   - Gathers recent completed candles (from request, database, or Binance fallback).
   - Validates that >= 60 completed candles exist to avoid premature prediction.
   - Computes features via shared/features.py (identical feature pipeline as training).
   - Runs model inference to obtain predicted class and calibrated probabilities.
   - Saves prediction to PostgreSQL with status 'PENDING'.
   - Returns standardized prediction JSON response.
3. Exposes /health and /metrics for Prometheus monitoring.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Depends, Query, HTTPException, status
from starlette.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd
import joblib

from shared.config import (
    BASE_DIR,
    BINANCE_SYMBOL,
)
from shared.features import (
    FEATURE_COLUMNS,
    MIN_CANDLES_FOR_FEATURES,
    LABEL_INT_TO_NAME,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    compute_features,
)
from shared.database import get_db, init_db, record_prediction, MarketData, Prediction
from shared.schemas import PredictionResponse, PredictionProbabilities
from shared.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    MODEL_VERSION_GAUGE,
    HTTP_REQUESTS_TOTAL,
    get_metrics_response,
)
from shared.binance_client import BinanceClient

logger = logging.getLogger("inference_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

binance_client = BinanceClient()

# Active in-memory model container
active_model_state: Dict[str, Any] = {
    "model": None,
    "version": "unknown",
    "metadata": {},
    "is_loaded": False,
    "load_time": None,
}


def load_active_model() -> bool:
    """
    Loads or reloads the approved model artifact from disk without restarting the application.
    """
    if not MODEL_FILE.exists():
        logger.warning("No model artifact found at: %s", MODEL_FILE)
        active_model_state["is_loaded"] = False
        return False

    try:
        model = joblib.load(MODEL_FILE)
        metadata = {}
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)

        version = metadata.get("model_version", "1.0.0")
        active_model_state["model"] = model
        active_model_state["version"] = version
        active_model_state["metadata"] = metadata
        active_model_state["is_loaded"] = True
        active_model_state["load_time"] = int(time.time())

        # Update Prometheus gauge
        MODEL_VERSION_GAUGE.labels(version=version).set(1.0)
        logger.info("Successfully loaded active model version: %s", version)
        return True

    except Exception as exc:
        logger.error("Failed to load model artifact: %s", exc)
        active_model_state["is_loaded"] = False
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database tables and loads model weights on startup."""
    try:
        init_db()
    except Exception as e:
        logger.warning("Could not initialize DB on startup: %s", e)
    load_active_model()
    yield


app = FastAPI(
    title="CryptoPulse - Inference Microservice",
    description="Real-time 15-minute Bitcoin price movement classifier REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

INDEX_HTML_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def get_web_ui():
    """Serves the live interactive Web UI for real-time BTC movement predictions."""
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>CryptoPulse Web UI</h1><p>Template index.html not found.</p>")


class PredictRequest(BaseModel):
    """Optional manual payload. If candles omitted, service automatically queries latest market data."""
    symbol: Optional[str] = Field(default=BINANCE_SYMBOL, description="Trading pair symbol")
    candles: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of at least 60 completed 1-minute OHLCV candles"
    )


@app.get("/health", tags=["Health & Monitoring"])
def health_check():
    """Returns service health status and currently loaded model version."""
    HTTP_REQUESTS_TOTAL.labels(service="inference", method="GET", endpoint="/health", status="200").inc()
    return {
        "service": "inference",
        "status": "healthy" if active_model_state["is_loaded"] else "degraded",
        "model_loaded": active_model_state["is_loaded"],
        "model_version": active_model_state["version"],
        "model_load_time": active_model_state["load_time"],
        "symbol": BINANCE_SYMBOL,
    }


@app.get("/metrics", tags=["Health & Monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return get_metrics_response()


@app.post("/reload-model", tags=["Model Lifecycle"])
def reload_model():
    """Dynamic model reloading endpoint: pulls new model weights from disk without downtime."""
    HTTP_REQUESTS_TOTAL.labels(service="inference", method="POST", endpoint="/reload-model", status="200").inc()
    success = load_active_model()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model artifact from models directory."
        )
    return {
        "status": "success",
        "message": "Model reloaded successfully",
        "model_version": active_model_state["version"],
    }


@app.post("/retrain", tags=["Model Lifecycle"])
def trigger_retrain(
    rows: int = Query(default=100_000, ge=1000),
    version: Optional[str] = Query(default=None)
):
    """
    Triggers model retraining, increments version (e.g. v1.0.0 -> v1.1.0),
    registers to MLflow, and automatically hot-reloads the active inference model.
    """
    HTTP_REQUESTS_TOTAL.labels(service="inference", method="POST", endpoint="/retrain", status="200").inc()
    from shared.config import TRAINING_SERVICE_URL
    import requests

    payload = {"sample_rows": rows}
    if version:
        payload["model_version"] = version

    try:
        resp = requests.post(f"{TRAINING_SERVICE_URL}/train", json=payload, timeout=300)
        if resp.ok:
            load_active_model()
            return resp.json()
        else:
            err_detail = resp.text
            try:
                err_detail = resp.json().get("detail", err_detail)
            except Exception:
                pass
            logger.warning("Training service error (%d): %s", resp.status_code, err_detail)
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Training service failed: {err_detail}"
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Could not reach training service at %s (%s). Attempting in-process retraining fallback...",
            TRAINING_SERVICE_URL,
            exc,
        )

    # Local fallback execution
    try:
        from services.training.train import train_model
        result = train_model(sample_rows=rows, model_version=version)
        load_active_model()
        return {
            "status": "success",
            "model_version": active_model_state["version"],
            "metrics": result["metrics"],
            "message": f"Retrained and switched to model version v{active_model_state['version']}"
        }
    except Exception as local_exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining execution failed: {local_exc}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_movement(
    request: Optional[PredictRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Predicts Bitcoin price movement over next 1 minute (UP, STABLE, or DOWN).
    Retrieves latest completed candles, computes 20 technical features, runs model inference,
    and stores the prediction record in PostgreSQL with status 'PENDING'.
    """
    if not active_model_state["is_loaded"]:
        # Attempt to load model if not yet loaded
        if not load_active_model():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model is not loaded. Train a model first via the training service."
            )

    start_timer = time.time()
    candles_list: List[Dict[str, Any]] = []

    # 1. Use candles provided in request, or fetch from DB / Binance
    if request and request.candles is not None:
        if len(request.candles) < MIN_CANDLES_FOR_FEATURES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Insufficient completed candles for feature engineering. "
                    f"Need at least {MIN_CANDLES_FOR_FEATURES} candles (e.g. for sma_60), "
                    f"but found {len(request.candles)}."
                )
            )
        candles_list = request.candles
    else:
        # Query recent candles from database
        try:
            db_records = (
                db.query(MarketData)
                .filter_by(symbol=BINANCE_SYMBOL)
                .order_by(MarketData.candle_timestamp.desc())
                .limit(MIN_CANDLES_FOR_FEATURES + 15)
                .all()
            )
        except Exception as db_err:
            logger.warning("Database query failed; falling back to Binance directly: %s", db_err)
            db_records = []

        if len(db_records) >= MIN_CANDLES_FOR_FEATURES:
            candles_list = [r.to_dict() for r in reversed(db_records)]
        else:
            # Fallback to Binance client if database buffer is warming up
            logger.info("Database has only %d candles; fetching live completed candles from Binance...", len(db_records))
            try:
                candles_list = binance_client.fetch_completed_candles(limit=MIN_CANDLES_FOR_FEATURES + 15)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to retrieve market data from Binance: {exc}"
                )

    # 2. Strict validation: Must have at least MIN_CANDLES_FOR_FEATURES completed candles
    if len(candles_list) < MIN_CANDLES_FOR_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Insufficient completed candles for feature engineering. "
                f"Need at least {MIN_CANDLES_FOR_FEATURES} candles (e.g. for sma_60), "
                f"but found {len(candles_list)}."
            )
        )

    # 3. Compute Features using Shared Module
    candle_df = pd.DataFrame(candles_list)
    feat_df = compute_features(candle_df, drop_na=True)

    if len(feat_df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feature calculation resulted in zero valid rows after rolling indicators."
        )

    # Latest observation features
    latest_row = feat_df.iloc[-1]
    X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]])
    prediction_time = int(latest_row["timestamp"])

    # 4. Run Model Inference
    model = active_model_state["model"]
    pred_class_idx = int(model.predict(X_latest)[0])
    probabilities_raw = model.predict_proba(X_latest)[0]

    predicted_label = LABEL_INT_TO_NAME.get(pred_class_idx, "STABLE")
    prob_dict = {
        "DOWN": round(float(probabilities_raw[LABEL_DOWN]), 4),
        "STABLE": round(float(probabilities_raw[LABEL_STABLE]), 4),
        "UP": round(float(probabilities_raw[LABEL_UP]), 4),
    }

    # Record Prometheus metrics
    PREDICTIONS_TOTAL.labels(predicted_class=predicted_label).inc()
    latency = time.time() - start_timer
    PREDICTION_LATENCY_SECONDS.observe(latency)
    HTTP_REQUESTS_TOTAL.labels(service="inference", method="POST", endpoint="/predict", status="200").inc()

    # 5. Persist Prediction to PostgreSQL (Pending delayed 1-minute evaluation)
    prediction_id = None
    try:
        pred_record = record_prediction(db, {
            "prediction_timestamp": prediction_time,
            "symbol": BINANCE_SYMBOL,
            "predicted_class": predicted_label,
            "class_probabilities": prob_dict,
            "model_version": active_model_state["version"],
        })
        prediction_id = pred_record.id
    except Exception as db_exc:
        logger.warning("Could not persist prediction to database: %s", db_exc)

    return PredictionResponse(
        symbol=BINANCE_SYMBOL,
        prediction=predicted_label,
        probabilities=PredictionProbabilities(**prob_dict),
        model_version=active_model_state["version"],
        prediction_timestamp=prediction_time,
        prediction_id=prediction_id,
    )


@app.get("/predictions/recent", tags=["Inference"])
def get_recent_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Returns recent predictions stored in the database.
    Performs an evaluation pass on matured pending predictions so the Web UI
    receives real-time ground truth updates within 1 minute.
    """
    HTTP_REQUESTS_TOTAL.labels(service="inference", method="GET", endpoint="/predictions/recent", status="200").inc()

    # Inline check for matured pending predictions
    try:
        from services.monitoring.evaluator import evaluate_pending_predictions
        evaluate_pending_predictions(db)
    except Exception as eval_err:
        logger.debug("Inline delayed evaluation notice: %s", eval_err)

    records = (
        db.query(Prediction)
        .order_by(Prediction.prediction_timestamp.desc())
        .limit(limit)
        .all()
    )

    now = int(time.time())
    result = []
    for r in records:
        d = r.to_dict()
        # Add seconds remaining for pending predictions
        if d["evaluation_status"] == "PENDING":
            elapsed = now - d["prediction_timestamp"]
            d["seconds_remaining"] = max(0, 60 - elapsed)
        else:
            d["seconds_remaining"] = 0
        result.append(d)

    return result

