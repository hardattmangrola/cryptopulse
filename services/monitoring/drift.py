"""
CryptoPulse - Continuous Monitoring: Data Drift Detection Module
Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics
comparing live production market features against baseline training reference distributions.

Thresholds (Standard Banking & MLOps Convention):
- PSI < 0.10: No Drift (Distribution is stable)
- 0.10 <= PSI < 0.25: Moderate Drift (Monitor closely)
- PSI >= 0.25: Significant Drift (Action required: triggers automated retraining)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from shared.config import BASE_DIR, PSI_DRIFT_THRESHOLD
from shared.features import FEATURE_COLUMNS
from shared.database import SessionLocal, record_drift_metrics
from shared.metrics import FEATURE_DRIFT_PSI, DRIFT_ALERTS_TOTAL

logger = logging.getLogger("drift_detector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_REF_PATH = BASE_DIR / "models" / "reference_distribution.json"


def calculate_psi(
    actual_values: np.ndarray,
    bin_edges: List[float],
    expected_proportions: Optional[List[float]] = None,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) for a continuous variable given bin edges.

    Args:
        actual_values: Array of live production observations.
        bin_edges: Monotonically increasing bin edges (including -inf and +inf).
        expected_proportions: Target proportions for each bin (defaults to uniform 0.10 for deciles).
        epsilon: Smoothing parameter to avoid division by zero or log(0).

    Returns:
        float: Computed PSI value.
    """
    clean_values = actual_values[~np.isnan(actual_values)]
    if len(clean_values) == 0:
        return 0.0

    n_bins = len(bin_edges) - 1
    if expected_proportions is None:
        expected_proportions = [1.0 / n_bins] * n_bins

    # Count observed values in each bin
    counts, _ = np.histogram(clean_values, bins=bin_edges)
    actual_proportions = counts / len(clean_values)

    # Apply smoothing
    actual_smooth = (actual_proportions + epsilon) / (np.sum(actual_proportions) + epsilon * n_bins)
    expected_smooth = (np.array(expected_proportions) + epsilon) / (np.sum(expected_proportions) + epsilon * n_bins)

    # PSI = sum((Actual - Expected) * ln(Actual / Expected))
    psi_value = np.sum((actual_smooth - expected_smooth) * np.log(actual_smooth / expected_smooth))
    return float(max(0.0, psi_value))


def load_reference_distribution(path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads baseline decile distributions saved during model training."""
    target_path = path or DEFAULT_REF_PATH
    if not target_path.exists():
        logger.warning("Reference distribution file not found at: %s", target_path)
        return {}
    with open(target_path, "r") as f:
        return json.load(f)


def analyze_feature_drift(
    prod_df: pd.DataFrame,
    ref_distribution: Optional[Dict[str, Any]] = None,
    ref_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates drift across all 20 technical features comparing live production data
    with the baseline reference distributions.

    Returns:
        Dict containing:
        - timestamp: int
        - sample_size: int
        - feature_metrics: Dict[feature, {psi, ks_stat, p_value, status}]
        - max_psi: float
        - max_drift_feature: str
        - drifted_features: List[str]
        - drift_detected: bool (True if max_psi >= PSI_DRIFT_THRESHOLD)
    """
    if ref_distribution is None:
        ref_distribution = load_reference_distribution(ref_path)

    if not ref_distribution:
        logger.warning("No reference distribution available. Skipping drift calculation.")
        return {
            "timestamp": int(pd.Timestamp.now().timestamp()),
            "sample_size": len(prod_df),
            "feature_metrics": {},
            "max_psi": 0.0,
            "max_drift_feature": None,
            "drifted_features": [],
            "drift_detected": False,
        }

    feature_metrics: Dict[str, Any] = {}
    drifted_features: List[str] = []
    max_psi = 0.0
    max_feature = None

    for feat in FEATURE_COLUMNS:
        if feat not in prod_df.columns or feat not in ref_distribution:
            continue

        ref_info = ref_distribution[feat]
        quantiles_dict = ref_info.get("quantiles", {})
        if not quantiles_dict:
            continue

        # Construct 10 bins: [-inf, q0.1, q0.2, ..., q0.9, +inf]
        q_vals = [float(quantiles_dict[k]) for k in sorted(quantiles_dict.keys(), key=lambda x: float(x))]
        
        # Ensure bin edges are strictly increasing to satisfy np.histogram
        bin_edges = [-np.inf]
        prev_val = -np.inf
        for q in q_vals:
            if q <= prev_val:
                q = prev_val + 1e-6
            bin_edges.append(q)
            prev_val = q
        bin_edges.append(np.inf)

        prod_vals = prod_df[feat].to_numpy(dtype=float)
        psi_score = calculate_psi(prod_vals, bin_edges)

        # Approximate two-sample KS test using reference mean and std
        ref_mean = ref_info.get("mean", 0.0)
        ref_std = max(ref_info.get("std", 1.0), 1e-6)
        clean_prod = prod_vals[~np.isnan(prod_vals)]
        if len(clean_prod) > 2:
            synthetic_ref = np.random.normal(ref_mean, ref_std, size=min(len(clean_prod), 500))
            ks_res = stats.ks_2samp(clean_prod, synthetic_ref)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)
        else:
            ks_stat, p_val = 0.0, 1.0

        # Classify feature status
        if psi_score >= PSI_DRIFT_THRESHOLD:
            status = "SIGNIFICANT_DRIFT"
            drifted_features.append(feat)
        elif psi_score >= 0.10:
            status = "MODERATE_DRIFT"
        else:
            status = "NO_DRIFT"

        feature_metrics[feat] = {
            "psi": round(psi_score, 4),
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_val, 4),
            "drift_status": status,
        }

        # Update Prometheus gauge
        FEATURE_DRIFT_PSI.labels(feature=feat).set(psi_score)

        if psi_score > max_psi:
            max_psi = psi_score
            max_feature = feat

    drift_detected = (max_psi >= PSI_DRIFT_THRESHOLD)
    if drift_detected:
        DRIFT_ALERTS_TOTAL.inc()
        logger.warning(
            "Significant feature drift detected! Max PSI: %.4f on '%s' (Breached threshold: %.2f)",
            max_psi,
            max_feature,
            PSI_DRIFT_THRESHOLD,
        )

    return {
        "timestamp": int(pd.Timestamp.now().timestamp()),
        "sample_size": len(prod_df),
        "feature_metrics": feature_metrics,
        "max_psi": round(max_psi, 4),
        "max_drift_feature": max_feature,
        "drifted_features": drifted_features,
        "drift_detected": drift_detected,
    }


def persist_drift_results_to_db(drift_results: Dict[str, Any]) -> int:
    """Writes drift metrics for each feature into PostgreSQL drift_metrics table."""
    timestamp = drift_results.get("timestamp", int(pd.Timestamp.now().timestamp()))
    records = []
    for feat, metrics in drift_results.get("feature_metrics", {}).items():
        records.append({
            "timestamp": timestamp,
            "feature_name": feat,
            "drift_statistic": metrics["psi"],
            "drift_status": metrics["drift_status"],
        })

    if not records:
        return 0

    try:
        with SessionLocal() as db:
            saved = record_drift_metrics(db, records)
            return saved
    except Exception as exc:
        logger.warning("Could not persist drift records to database: %s", exc)
        return 0
