"""
CryptoPulse - Monitoring & Continuous Maintenance Microservice
Coordinates continuous 1-minute delayed ground-truth evaluation,
real-time feature drift detection (PSI & KS-test),
and automated retraining orchestration with cooldown safety.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, Depends, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from shared.config import (
    BINANCE_SYMBOL,
    EVALUATION_POLL_INTERVAL_SECONDS,
    DRIFT_CHECK_INTERVAL_SECONDS,
)
from shared.database import get_db, init_db, SessionLocal, MarketData, Prediction, PredictionOutcome
from shared.features import compute_features, MIN_CANDLES_FOR_FEATURES
from shared.metrics import HTTP_REQUESTS_TOTAL, get_metrics_response
from services.monitoring.evaluator import evaluate_pending_predictions, calculate_rolling_performance
from services.monitoring.drift import analyze_feature_drift, persist_drift_results_to_db
from services.monitoring.controller import controller

logger = logging.getLogger("monitoring_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global monitoring state
monitoring_state: Dict[str, Any] = {
    "is_running": False,
    "last_evaluation_time": None,
    "last_evaluation_summary": None,
    "last_drift_time": None,
    "last_drift_summary": None,
    "last_retrain_time": None,
    "last_retrain_summary": None,
}

_stop_event = asyncio.Event()


async def background_monitoring_loop():
    """Continuous background loop for delayed 1-minute evaluation and drift monitoring."""
    logger.info("Starting background continuous monitoring loop...")
    monitoring_state["is_running"] = True
    last_drift_check = 0.0

    while not _stop_event.is_set():
        try:
            # 1. Delayed 1-minute evaluation (runs every ~5 seconds)
            with SessionLocal() as db:
                eval_summary = evaluate_pending_predictions(db)
                monitoring_state["last_evaluation_time"] = int(time.time())
                monitoring_state["last_evaluation_summary"] = eval_summary

            # 2. Drift check & Retraining check (runs every DRIFT_CHECK_INTERVAL_SECONDS)
            now = time.time()
            if now - last_drift_check >= DRIFT_CHECK_INTERVAL_SECONDS:
                last_drift_check = now
                with SessionLocal() as db:
                    # Query recent candles to compute live features
                    recent_candles = (
                        db.query(MarketData)
                        .filter_by(symbol=BINANCE_SYMBOL)
                        .order_by(MarketData.candle_timestamp.desc())
                        .limit(MIN_CANDLES_FOR_FEATURES + 100)
                        .all()
                    )
                    if len(recent_candles) >= MIN_CANDLES_FOR_FEATURES:
                        candle_df = pd.DataFrame([c.to_dict() for c in reversed(recent_candles)])
                        feat_df = compute_features(candle_df, drop_na=True)
                        if len(feat_df) > 0:
                            drift_res = analyze_feature_drift(feat_df)
                            persist_drift_results_to_db(drift_res)
                            monitoring_state["last_drift_time"] = int(now)
                            monitoring_state["last_drift_summary"] = drift_res

                            # Retraining check
                            rolling_metrics = eval_summary.get("rolling_metrics")
                            should_trigger, reason = controller.evaluate_retraining_trigger(
                                drift_results=drift_res,
                                rolling_metrics=rolling_metrics,
                            )
                            if should_trigger:
                                logger.info("Retraining trigger met: %s. Initiating controller...", reason)
                                retrain_res = controller.trigger_retraining(reason=reason)
                                monitoring_state["last_retrain_summary"] = retrain_res

        except Exception as exc:
            logger.error("Background monitoring loop error: %s", exc)

        # Sleep before next cycle
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=EVALUATION_POLL_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes tables and starts background evaluation worker on startup."""
    try:
        init_db()
    except Exception as e:
        logger.warning("Could not initialize DB on startup: %s", e)

    task = asyncio.create_task(background_monitoring_loop())
    yield
    _stop_event.set()
    await task


app = FastAPI(
    title="CryptoPulse - Monitoring & Continuous Maintenance Service",
    description="Manages delayed 1-minute performance evaluation, feature drift (PSI/KS), and retraining controller.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health & Monitoring"])
def health_check():
    """Returns service health status."""
    HTTP_REQUESTS_TOTAL.labels(service="monitoring", method="GET", endpoint="/health", status="200").inc()
    return {
        "service": "monitoring",
        "status": "healthy" if monitoring_state["is_running"] else "idle",
        "last_evaluation_time": monitoring_state["last_evaluation_time"],
        "last_drift_time": monitoring_state["last_drift_time"],
    }


@app.get("/metrics", tags=["Health & Monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return get_metrics_response()


@app.post("/evaluate", tags=["Evaluation"])
def trigger_evaluation(db: Session = Depends(get_db)):
    """Triggers immediate evaluation of matured pending 1-minute predictions."""
    HTTP_REQUESTS_TOTAL.labels(service="monitoring", method="POST", endpoint="/evaluate", status="200").inc()
    summary = evaluate_pending_predictions(db)
    monitoring_state["last_evaluation_time"] = int(time.time())
    monitoring_state["last_evaluation_summary"] = summary
    return summary


@app.post("/drift/check", tags=["Drift Detection"])
def trigger_drift_check(db: Session = Depends(get_db)):
    """Computes PSI and KS drift metrics on recent production features."""
    HTTP_REQUESTS_TOTAL.labels(service="monitoring", method="POST", endpoint="/drift/check", status="200").inc()
    recent_candles = (
        db.query(MarketData)
        .filter_by(symbol=BINANCE_SYMBOL)
        .order_by(MarketData.candle_timestamp.desc())
        .limit(MIN_CANDLES_FOR_FEATURES + 100)
        .all()
    )

    if len(recent_candles) < MIN_CANDLES_FOR_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_CANDLES_FOR_FEATURES} candles in database to compute features for drift check."
        )

    candle_df = pd.DataFrame([c.to_dict() for c in reversed(recent_candles)])
    feat_df = compute_features(candle_df, drop_na=True)
    drift_res = analyze_feature_drift(feat_df)
    persist_drift_results_to_db(drift_res)

    monitoring_state["last_drift_time"] = int(time.time())
    monitoring_state["last_drift_summary"] = drift_res
    return drift_res


class RetrainRequest(BaseModel):
    reason: Optional[str] = Field(default="manual_trigger", description="Reason for triggering retraining")
    force: bool = Field(default=False, description="Bypass cooldown check")
    sample_rows: int = Field(default=100_000, ge=1000)


@app.post("/controller/retrain", tags=["Retraining Controller"])
def trigger_controller_retraining(req: Optional[RetrainRequest] = None):
    """Triggers retraining through the controller, respecting cooldown unless forced."""
    HTTP_REQUESTS_TOTAL.labels(service="monitoring", method="POST", endpoint="/controller/retrain", status="200").inc()
    params = req or RetrainRequest()
    res = controller.trigger_retraining(
        reason=params.reason,
        sample_rows=params.sample_rows,
        force=params.force,
    )
    monitoring_state["last_retrain_summary"] = res
    return res


@app.get("/status", tags=["Status & Dashboard"])
def get_monitoring_dashboard_status(db: Session = Depends(get_db)):
    """Consolidated status endpoint for Web UI and dashboard reporting."""
    HTTP_REQUESTS_TOTAL.labels(service="monitoring", method="GET", endpoint="/status", status="200").inc()
    total_preds = db.query(Prediction).count()
    pending_preds = db.query(Prediction).filter_by(evaluation_status="PENDING").count()
    evaluated_preds = db.query(Prediction).filter_by(evaluation_status="EVALUATED").count()
    correct_preds = (
        db.query(PredictionOutcome)
        .filter_by(is_correct=True)
        .count()
    )

    in_cooldown, cooldown_remaining = controller.is_in_cooldown()
    rolling_metrics = calculate_rolling_performance(db, window=50)

    return {
        "status": "active",
        "predictions": {
            "total": total_preds,
            "pending": pending_preds,
            "evaluated": evaluated_preds,
            "correct": correct_preds,
            "accuracy": (correct_preds / evaluated_preds) if evaluated_preds > 0 else None,
        },
        "rolling_metrics": rolling_metrics,
        "drift": monitoring_state.get("last_drift_summary"),
        "controller": {
            "in_cooldown": in_cooldown,
            "cooldown_remaining_seconds": cooldown_remaining,
            "last_trigger_reason": controller.last_trigger_reason,
            "last_result": controller.last_run_result,
        },
    }
