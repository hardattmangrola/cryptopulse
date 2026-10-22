"""
CryptoPulse - Model Training Pipeline
Trains an XGBoost 3-class classification model to predict 15-minute Bitcoin price direction.

Workflow:
1. Load historical BTC/USD 1-minute market data.
2. Validate and sort timestamps chronologically.
3. Generate reusable technical features via shared/features.py (zero lookahead).
4. Generate 3-class target labels (DOWN=0, STABLE=1, UP=2) based on 15m return.
5. Chronologically split into train (70%), validation (15%), and test (15%) sets without shuffling.
6. Train XGBoost model with multi:softprob objective.
7. Evaluate using Macro F1 (primary), Accuracy, Precision, Recall, and Confusion Matrix.
8. Extract reference feature distributions for drift monitoring.
9. Persist model artifact, metrics, and distribution to disk and log to MLflow.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.xgboost

from shared.config import (
    BASE_DIR,
    HISTORICAL_DATA_PATH,
)
from shared.data_loader import validate_csv_file
from shared.features import (
    FEATURE_COLUMNS,
    DEFAULT_PREDICTION_HORIZON_MINUTES,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    LABEL_INT_TO_NAME,
    LABEL_DOWN,
    LABEL_STABLE,
    LABEL_UP,
    prepare_training_features_and_targets,
)

# Logger setup
logger = logging.getLogger("train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Artifacts output directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data(
    file_path: Path,
    sample_rows: Optional[int] = 100_000
) -> pd.DataFrame:
    """
    Loads historical market candles for model training.

    For realistic university execution speed and memory efficiency, defaults to the
    most recent `sample_rows` records (which represents ~70 days of recent BTC trading).
    Pass sample_rows=None to train on the entire historical dataset.

    Args:
        file_path: Path to btcusd_1-min_data.csv.
        sample_rows: Number of most recent rows to load (default: 100,000).

    Returns:
        pd.DataFrame: Sorted, validated raw OHLCV DataFrame.
    """
    validate_csv_file(file_path)
    logger.info("Reading raw market data from: %s", file_path)

    if sample_rows is not None and sample_rows > 0:
        # Count total rows to read the most recent sample_rows from end of file
        with open(file_path, "rb") as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header
        skip_count = max(0, total_lines - sample_rows)
        logger.info(
            "Loading most recent %d rows (skipping initial %d rows of %d total)...",
            sample_rows,
            skip_count,
            total_lines,
        )
        if skip_count > 0:
            df = pd.read_csv(file_path, skiprows=range(1, skip_count + 1))
        else:
            df = pd.read_csv(file_path)
    else:
        logger.info("Loading full historical dataset...")
        df = pd.read_csv(file_path)

    # Sort strictly chronologically
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    logger.info("Loaded %d raw candles spanning timestamps %d to %d.", len(df), df["Timestamp"].iloc[0], df["Timestamp"].iloc[-1])
    return df


def split_chronological(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits time series data strictly chronologically without random shuffling.
    Prevents temporal lookahead and data leakage.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(
        "Chronological Split: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Test=%d (%.1f%%)",
        len(X_train),
        train_ratio * 100,
        len(X_val),
        val_ratio * 100,
        len(X_test),
        (1 - train_ratio - val_ratio) * 100,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def calculate_feature_reference_distribution(X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates reference distributions (mean, std, percentiles) for each feature on training data.
    These statistics are stored and used by the Monitoring service to detect production data drift (PSI/KS).
    """
    ref_dist: Dict[str, Any] = {}
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for col in FEATURE_COLUMNS:
        series = X_train[col].dropna()
        q_values = series.quantile(quantiles).to_dict()
        ref_dist[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "quantiles": {str(k): float(v) for k, v in q_values.items()},
        }
    return ref_dist


def evaluate_model(
    model: XGBClassifier,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    split_name: str = "Test"
) -> Dict[str, Any]:
    """
    Computes classification metrics for a given evaluation split.
    Uses Macro F1 as the primary metric.
    """
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)

    macro_f1 = float(f1_score(y_eval, preds, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_eval, preds))
    macro_precision = float(precision_score(y_eval, preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_eval, preds, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_eval, preds, labels=[LABEL_DOWN, LABEL_STABLE, LABEL_UP]).tolist()

    class_f1 = f1_score(y_eval, preds, average=None, labels=[LABEL_DOWN, LABEL_STABLE, LABEL_UP], zero_division=0)
    per_class_metrics = {
        "DOWN_F1": float(class_f1[0]),
        "STABLE_F1": float(class_f1[1]),
        "UP_F1": float(class_f1[2]),
    }

    metrics = {
        f"{split_name.lower()}_macro_f1": macro_f1,
        f"{split_name.lower()}_accuracy": accuracy,
        f"{split_name.lower()}_macro_precision": macro_precision,
        f"{split_name.lower()}_macro_recall": macro_recall,
        f"{split_name.lower()}_per_class_f1": per_class_metrics,
        f"{split_name.lower()}_confusion_matrix": conf_mat,
    }

    logger.info(
        "[%s Metrics] Macro F1: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f",
        split_name,
        macro_f1,
        accuracy,
        macro_precision,
        macro_recall,
    )
    return metrics


def get_next_model_version(explicit_version: Optional[str] = None) -> str:
    """
    Determines the next incremental model version.
    If explicit_version is passed, returns it.
    Otherwise, reads models/model_metadata.json and increments: e.g. 1.0.0 -> 1.1.0.
    Defaults to 1.0.0 if no prior metadata exists.
    """
    if explicit_version and explicit_version.strip():
        return explicit_version.strip()

    metadata_path = MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                current_meta = json.load(f)
            curr_ver = current_meta.get("model_version", "1.0.0")
            parts = curr_ver.split(".")
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{major}.{minor + 1}.0"
        except Exception as e:
            logger.debug("Could not parse current model version for increment: %s", e)
    return "1.0.0"


def train_model(
    sample_rows: Optional[int] = 100_000,
    horizon_minutes: int = DEFAULT_PREDICTION_HORIZON_MINUTES,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    mlflow_tracking_uri: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the full model training and artifact persistence workflow.
    """
    resolved_version = get_next_model_version(model_version)
    logger.info("=== Starting CryptoPulse Model Training Pipeline (Target Version: v%s) ===", resolved_version)

    # 1. Load historical raw dataset
    raw_df = load_training_data(HISTORICAL_DATA_PATH, sample_rows=sample_rows)

    # 2. Extract features and future 3-class target labels
    logger.info("Generating features and future target labels (horizon=%dm, threshold=%.4f)...", horizon_minutes, threshold)
    X, y, clean_df = prepare_training_features_and_targets(
        raw_df, horizon_minutes=horizon_minutes, threshold=threshold
    )
    logger.info("Engineered feature matrix: %s, target length: %d", X.shape, len(y))

    # Log class distributions
    class_dist = y.value_counts().to_dict()
    class_dist_named = {LABEL_INT_TO_NAME[k]: v for k, v in class_dist.items()}
    logger.info("Target class distribution: %s", class_dist_named)

    # 3. Chronological train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = split_chronological(X, y)

    # 4. Initialize and fit XGBoost Classifier
    logger.info("Training XGBoost Classifier (n_estimators=%d, max_depth=%d, lr=%.2f)...", n_estimators, max_depth, learning_rate)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",  # Fast histogram-based method
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 5. Evaluate on Validation and Test sets
    val_metrics = evaluate_model(model, X_val, y_val, split_name="Val")
    test_metrics = evaluate_model(model, X_test, y_test, split_name="Test")

    all_metrics = {**val_metrics, **test_metrics}

    # 6. Extract Reference Distribution on Training Data (for Drift Detection in Phase 7)
    logger.info("Computing reference feature distributions for production drift monitoring...")
    ref_dist = calculate_feature_reference_distribution(X_train)

    # 7. Model Metadata
    metadata = {
        "model_version": resolved_version,
        "model_type": "XGBClassifier",
        "prediction_horizon_minutes": horizon_minutes,
        "classification_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": LABEL_INT_TO_NAME,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "dataset_rows_used": len(raw_df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "metrics": all_metrics,
    }

    # 8. Save artifacts to models/ directory
    model_artifact_path = MODELS_DIR / "baseline_model.joblib"
    metrics_path = MODELS_DIR / "metrics.json"
    ref_dist_path = MODELS_DIR / "reference_distribution.json"
    metadata_path = MODELS_DIR / "model_metadata.json"

    logger.info("Saving model artifact to: %s", model_artifact_path)
    joblib.dump(model, model_artifact_path)

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(ref_dist_path, "w") as f:
        json.dump(ref_dist, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metrics, metadata, and reference distributions to %s", MODELS_DIR)

    # 9. Log to MLflow
    try:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("cryptopulse_btc_movement")

        run_name = f"cryptopulse_v{resolved_version}"
        with mlflow.start_run(run_name=run_name) as run:
            # Set metadata tags in MLflow
            mlflow.set_tag("model_version", resolved_version)
            mlflow.set_tag("horizon", f"{horizon_minutes}m")
            mlflow.set_tag("registered_model", "cryptopulse_btc_movement")

            # Log params
            mlflow.log_params({
                "model_version": resolved_version,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "horizon_minutes": horizon_minutes,
                "threshold": threshold,
                "sample_rows": sample_rows,
            })
            # Log key metrics
            mlflow.log_metrics({
                "val_macro_f1": val_metrics["val_macro_f1"],
                "val_accuracy": val_metrics["val_accuracy"],
                "test_macro_f1": test_metrics["test_macro_f1"],
                "test_accuracy": test_metrics["test_accuracy"],
                "test_macro_precision": test_metrics["test_macro_precision"],
                "test_macro_recall": test_metrics["test_macro_recall"],
            })
            # Log model artifact & register to MLflow Model Registry
            try:
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="cryptopulse_btc_movement",
                )
            except Exception as ml_m_err:
                logger.info("mlflow.xgboost.log_model notice: %s", ml_m_err)

            mlflow.log_artifact(str(model_artifact_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(metadata_path))
            run_id = run.info.run_id
            logger.info("Experiment run successfully logged and registered to MLflow (Run ID: %s, Version: v%s)", run_id, resolved_version)

    except Exception as exc:
        run_id = None
        logger.warning("MLflow logging encountered notice (continuing with local artifacts): %s", exc)

    # 10. Persist Model Run metadata to database
    try:
        from shared.database import SessionLocal, record_model_run, init_db
        init_db()
        with SessionLocal() as db:
            record_model_run(db, {
                "model_version": metadata["model_version"],
                "dataset_description": f"Historical BTC/USD 1m (last {metadata['dataset_rows_used']} rows)",
                "accuracy": test_metrics["test_accuracy"],
                "macro_f1": test_metrics["test_macro_f1"],
                "training_status": "COMPLETED",
                "promotion_status": "PROMOTED",
                "mlflow_run_id": run_id,
                "parameters": metadata["hyperparameters"],
                "metrics": all_metrics,
            })
            logger.info("Model run metadata persisted to database.")
    except Exception as db_exc:
        logger.info("Database persistence notice (offline or waiting for PostgreSQL container): %s", db_exc)

    logger.info("=== Model Training Finished Successfully for Version v%s ===", resolved_version)
    return {
        "model_path": str(model_artifact_path),
        "metrics": all_metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    # Command line entry point: allow optional row count and version overrides
    import argparse
    parser = argparse.ArgumentParser(description="Train CryptoPulse Baseline Model")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of recent rows to train on (default: 100000)")
    parser.add_argument("--version", type=str, default=None, help="Explicit model version (e.g. 1.1.0). If omitted, increments automatically.")
    args = parser.parse_args()

    train_model(sample_rows=args.rows, model_version=args.version)
