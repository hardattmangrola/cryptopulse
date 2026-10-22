"""
CryptoPulse - Continuous Monitoring: Delayed Performance Evaluator
Evaluates predictions against delayed ground-truth market outcomes at t + 1m (60 seconds).

Workflow:
1. Queries predictions from PostgreSQL where evaluation_status == 'PENDING'.
2. Checks if 60 seconds have elapsed since prediction_timestamp.
3. Retrieves actual candle close at t + 60s (from database or Binance).
4. Computes actual return: R = (Close_{t+60s} - Close_t) / Close_t.
5. Assigns ground truth class (UP if R > +0.05%, DOWN if R < -0.05%, STABLE otherwise).
6. Determines correctness (is_correct = predicted_class == actual_class).
7. Persists evaluation to prediction_outcomes table and marks prediction as EVALUATED.
8. Computes rolling Macro F1, Accuracy, and exports to Prometheus.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from shared.config import (
    BINANCE_SYMBOL,
    PREDICTION_HORIZON_SECONDS,
    CLASSIFICATION_THRESHOLD,
)
from shared.database import (
    SessionLocal,
    Prediction,
    PredictionOutcome,
    MarketData,
    record_prediction_outcome,
)
from shared.features import (
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    LABEL_NAME_TO_INT,
)
from shared.metrics import (
    PREDICTION_OUTCOMES_TOTAL,
    MODEL_ACCURACY_ROLLING,
    MODEL_MACRO_F1_ROLLING,
)
from shared.binance_client import BinanceClient

logger = logging.getLogger("performance_evaluator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

binance_client = BinanceClient()


def classify_return(actual_return: float, threshold: float = CLASSIFICATION_THRESHOLD) -> str:
    """Classifies percentage price return into UP, STABLE, or DOWN."""
    if actual_return > threshold:
        return "UP"
    elif actual_return < -threshold:
        return "DOWN"
    return "STABLE"


def find_candle_close(db: Session, symbol: str, target_timestamp: int) -> Optional[float]:
    """
    Finds the close price of a candle near the given timestamp.
    Searches PostgreSQL first, falling back to Binance public API if missing.
    """
    # 1. Search database within +/- 30 seconds
    candle = (
        db.query(MarketData)
        .filter(MarketData.symbol == symbol)
        .filter(MarketData.candle_timestamp >= target_timestamp - 30)
        .filter(MarketData.candle_timestamp <= target_timestamp + 30)
        .order_by(MarketData.candle_timestamp.asc())
        .first()
    )
    if candle:
        return float(candle.close)

    # 2. Query Binance klines directly for historical candle
    try:
        candles = binance_client.fetch_klines(
            symbol=symbol,
            interval="1m",
            start_time=target_timestamp - 60,
            end_time=target_timestamp + 60,
            limit=5,
        )
        if candles:
            # Pick closest candle
            closest = min(candles, key=lambda c: abs(c["timestamp"] - target_timestamp))
            return float(closest["close"])
    except Exception as exc:
        logger.debug("Binance fallback query notice for timestamp %d: %s", target_timestamp, exc)

    return None


def evaluate_pending_predictions(
    db: Session,
    max_batch: int = 50,
    horizon_seconds: int = PREDICTION_HORIZON_SECONDS,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Scans for pending predictions that have matured (current_time >= prediction_timestamp + horizon_seconds).
    Computes ground truth, records outcome, and updates status.

    Returns:
        Summary dict of evaluated batch and rolling metrics.
    """
    now = int(time.time())
    maturity_cutoff = now - horizon_seconds

    # Query pending predictions older than maturity cutoff
    pending = (
        db.query(Prediction)
        .filter(Prediction.evaluation_status == "PENDING")
        .filter(Prediction.prediction_timestamp <= maturity_cutoff)
        .order_by(Prediction.prediction_timestamp.asc())
        .limit(max_batch)
        .all()
    )

    evaluated_count = 0
    correct_count = 0

    for pred in pending:
        t0 = pred.prediction_timestamp
        t1 = t0 + horizon_seconds

        # Get initial close price P0 and future close price P1
        p0 = find_candle_close(db, pred.symbol, t0)
        p1 = find_candle_close(db, pred.symbol, t1)

        # If future candle not yet available, skip and retry next cycle
        if p0 is None or p1 is None or p0 <= 0:
            continue

        actual_return = (p1 - p0) / p0
        actual_class = classify_return(actual_return, threshold)
        is_correct = (pred.predicted_class == actual_class)

        outcome_data = {
            "prediction_id": pred.id,
            "evaluation_timestamp": now,
            "future_close": p1,
            "actual_return": round(actual_return, 6),
            "actual_class": actual_class,
            "is_correct": is_correct,
        }

        try:
            record_prediction_outcome(db, outcome_data)
            evaluated_count += 1
            if is_correct:
                correct_count += 1

            # Update Prometheus outcome counter
            PREDICTION_OUTCOMES_TOTAL.labels(
                actual_class=actual_class,
                is_correct=str(is_correct).lower()
            ).inc()

            logger.info(
                "Evaluated Prediction #%d: Predicted=%s, Actual=%s (Return=%+.4f%%) -> %s",
                pred.id,
                pred.predicted_class,
                actual_class,
                actual_return * 100,
                "CORRECT" if is_correct else "INCORRECT",
            )
        except Exception as exc:
            db.rollback()
            logger.warning("Could not record outcome for prediction #%d: %s", pred.id, exc)

    # Calculate rolling metrics on recent outcomes
    rolling_metrics = calculate_rolling_performance(db, window=100)

    return {
        "matured_pending_found": len(pending),
        "evaluated_count": evaluated_count,
        "correct_count": correct_count,
        "batch_accuracy": (correct_count / evaluated_count) if evaluated_count > 0 else None,
        "rolling_metrics": rolling_metrics,
    }


def calculate_rolling_performance(db: Session, window: int = 100) -> Dict[str, Any]:
    """
    Computes rolling Macro F1, Accuracy, Precision, and Recall across the last `window` outcomes.
    Updates Prometheus gauges.
    """
    recent_outcomes = (
        db.query(PredictionOutcome, Prediction)
        .join(Prediction, PredictionOutcome.prediction_id == Prediction.id)
        .order_by(PredictionOutcome.evaluation_timestamp.desc())
        .limit(window)
        .all()
    )

    if not recent_outcomes or len(recent_outcomes) < 3:
        return {
            "sample_size": len(recent_outcomes),
            "rolling_accuracy": None,
            "rolling_macro_f1": None,
        }

    y_true = []
    y_pred = []
    for outcome, pred in recent_outcomes:
        y_true.append(LABEL_NAME_TO_INT.get(outcome.actual_class, LABEL_STABLE))
        y_pred.append(LABEL_NAME_TO_INT.get(pred.predicted_class, LABEL_STABLE))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Update Prometheus gauges
    MODEL_ACCURACY_ROLLING.set(acc)
    MODEL_MACRO_F1_ROLLING.set(macro_f1)

    return {
        "sample_size": len(recent_outcomes),
        "rolling_accuracy": round(acc, 4),
        "rolling_macro_f1": round(macro_f1, 4),
        "rolling_macro_precision": round(macro_prec, 4),
        "rolling_macro_recall": round(macro_rec, 4),
    }
