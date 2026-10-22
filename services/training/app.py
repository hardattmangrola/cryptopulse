"""
CryptoPulse - Training Microservice
Provides REST endpoints and background execution for model training, candidate evaluation,
quality gate enforcement, and MLflow experiment logging.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Depends, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from shared.config import BASE_DIR, MLFLOW_TRACKING_URI
from shared.database import get_db, init_db, ModelRun
from shared.metrics import (
    TRAINING_RUNS_TOTAL,
    MODEL_TEST_MACRO_F1,
    HTTP_REQUESTS_TOTAL,
    get_metrics_response,
)
from services.training.train import train_model, MODELS_DIR

logger = logging.getLogger("training_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="CryptoPulse - Training Microservice",
    description="Manages model training, candidate evaluation, quality gates, and MLflow logging.",
    version="1.0.0",
)

# Current training state
training_state: Dict[str, Any] = {
    "is_training": False,
    "last_run_status": "none",
    "last_run_version": "1.0.0",
    "last_error": None,
    "last_metrics": None,
}


class TrainRequest(BaseModel):
    sample_rows: int = Field(default=100_000, ge=1000, description="Recent market rows to train on")
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: int = Field(default=5, ge=2, le=10)
    learning_rate: float = Field(default=0.05, ge=0.001, le=0.5)
    model_version: Optional[str] = Field(default=None, description="Optional target version (e.g. 1.1.0). If omitted, increments automatically.")
    quality_gate_f1_threshold: float = Field(
        default=0.30,
        description="Minimum Macro F1 required for candidate model promotion"
    )


def execute_training_with_quality_gate(
    params: TrainRequest,
    db: Session
) -> Dict[str, Any]:
    """
    Executes training and applies model quality gate before promotion.
    Preserves existing production model if candidate fails gate.
    """
    training_state["is_training"] = True
    try:
        # 1. Train candidate model
        result = train_model(
            sample_rows=params.sample_rows,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            model_version=params.model_version,
        )

        test_macro_f1 = result["metrics"]["test_macro_f1"]
        candidate_version = result["metadata"]["model_version"]

        # 2. Quality Gate Check
        passed_quality_gate = test_macro_f1 >= params.quality_gate_f1_threshold
        promotion_status = "PROMOTED" if passed_quality_gate else "REJECTED"

        # Update database record with quality gate decision
        run_record = db.query(ModelRun).filter_by(model_version=candidate_version).first()
        if run_record:
            run_record.promotion_status = promotion_status
            db.commit()

        # Update Prometheus metrics
        TRAINING_RUNS_TOTAL.labels(status="success").inc()
        if passed_quality_gate:
            MODEL_TEST_MACRO_F1.set(test_macro_f1)
            # Instruct inference service to hot-reload approved model
            try:
                import requests
                from shared.config import INFERENCE_SERVICE_URL
                requests.post(f"{INFERENCE_SERVICE_URL}/reload-model", timeout=5)
                logger.info("Notified inference service to reload approved model v%s", candidate_version)
            except Exception as rel_err:
                logger.debug("Inference reload notice: %s", rel_err)

        training_state["is_training"] = False
        training_state["last_run_status"] = "success"
        training_state["last_run_version"] = candidate_version
        training_state["last_metrics"] = result["metrics"]
        training_state["last_error"] = None

        logger.info(
            "Training complete. Candidate version %s: Test Macro F1=%.4f -> Quality Gate: %s",
            candidate_version,
            test_macro_f1,
            promotion_status,
        )

        return {
            "status": "success",
            "model_version": candidate_version,
            "metrics": result["metrics"],
            "quality_gate": {
                "threshold": params.quality_gate_f1_threshold,
                "candidate_macro_f1": test_macro_f1,
                "passed": passed_quality_gate,
                "promotion_status": promotion_status,
            },
        }

    except Exception as exc:
        TRAINING_RUNS_TOTAL.labels(status="failed").inc()
        training_state["is_training"] = False
        training_state["last_run_status"] = "failed"
        training_state["last_error"] = str(exc)
        logger.exception("Model training run failed: %s", exc)
        return {"status": "error", "error": str(exc)}


@app.get("/health", tags=["Health & Monitoring"])
def health_check():
    """Returns training service health, active status, and last run result."""
    HTTP_REQUESTS_TOTAL.labels(service="training", method="GET", endpoint="/health", status="200").inc()
    return {
        "service": "training",
        "status": "training" if training_state["is_training"] else "idle",
        "last_run_status": training_state["last_run_status"],
        "last_run_version": training_state["last_run_version"],
        "last_metrics": training_state["last_metrics"],
        "last_error": training_state["last_error"],
    }


@app.get("/metrics", tags=["Health & Monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return get_metrics_response()


@app.post("/train", tags=["Model Training"])
def trigger_training(
    request: Optional[TrainRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Triggers model training on historical / accumulated market data,
    evaluates Macro F1, and enforces the candidate quality gate.
    """
    if training_state["is_training"]:
        raise HTTPException(
            status_code=409,
            detail="A training job is already currently in progress."
        )

    HTTP_REQUESTS_TOTAL.labels(service="training", method="POST", endpoint="/train", status="200").inc()
    params = request or TrainRequest()
    result = execute_training_with_quality_gate(params, db)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return result


@app.get("/model-runs", tags=["Model History"])
def get_model_runs(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Retrieves model training run history from the database."""
    HTTP_REQUESTS_TOTAL.labels(service="training", method="GET", endpoint="/model-runs", status="200").inc()
    runs = db.query(ModelRun).order_by(ModelRun.id.desc()).limit(limit).all()
    return [r.to_dict() for r in runs]
