"""
CryptoPulse - Continuous Monitoring: Retraining Controller
Automates model retraining triggers when feature drift or model degradation is detected,
safeguarded by an enforced cooldown window and quality gate verification.

Trigger Conditions:
1. Significant Feature Drift: Any feature PSI >= 0.25
2. Performance Degradation: Rolling Macro F1 < 0.30
3. Cooldown Enforcement: At least 10 minutes (600s) between retraining triggers.
"""

import time
import logging
import requests
from typing import Dict, Any, Optional, Tuple

from shared.config import (
    TRAINING_SERVICE_URL,
    INFERENCE_SERVICE_URL,
    RETRAINING_COOLDOWN_MINUTES,
    RETRAINING_MIN_MACRO_F1,
    PSI_DRIFT_THRESHOLD,
)
from shared.metrics import RETRAINING_TRIGGERED_TOTAL

logger = logging.getLogger("retraining_controller")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class RetrainingController:
    """Manages retraining checks, cooldown logic, and service orchestration."""

    def __init__(
        self,
        cooldown_minutes: int = RETRAINING_COOLDOWN_MINUTES,
        min_macro_f1: float = RETRAINING_MIN_MACRO_F1,
        training_url: str = TRAINING_SERVICE_URL,
        inference_url: str = INFERENCE_SERVICE_URL,
    ):
        self.cooldown_seconds = cooldown_minutes * 60
        self.min_macro_f1 = min_macro_f1
        self.training_url = training_url.rstrip("/")
        self.inference_url = inference_url.rstrip("/")
        self.last_retrain_time: Optional[float] = None
        self.last_trigger_reason: Optional[str] = None
        self.last_run_result: Optional[Dict[str, Any]] = None

    def is_in_cooldown(self) -> Tuple[bool, int]:
        """Returns (in_cooldown: bool, seconds_remaining: int)."""
        if self.last_retrain_time is None:
            return False, 0
        elapsed = time.time() - self.last_retrain_time
        if elapsed < self.cooldown_seconds:
            return True, int(self.cooldown_seconds - elapsed)
        return False, 0

    def evaluate_retraining_trigger(
        self,
        drift_results: Optional[Dict[str, Any]] = None,
        rolling_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Determines whether automated retraining should be triggered.
        Returns (should_trigger: bool, reason: str).
        """
        # 1. Check feature drift trigger
        if drift_results and drift_results.get("drift_detected"):
            max_psi = drift_results.get("max_psi", 0.0)
            feat = drift_results.get("max_drift_feature", "unknown")
            return True, f"drift: {feat} PSI={max_psi:.4f} >= {PSI_DRIFT_THRESHOLD}"

        # 2. Check performance degradation trigger
        if rolling_metrics:
            sample_size = rolling_metrics.get("sample_size", 0)
            f1 = rolling_metrics.get("rolling_macro_f1")
            if sample_size >= 10 and f1 is not None and f1 < self.min_macro_f1:
                return True, f"performance_degradation: Rolling Macro F1={f1:.4f} < {self.min_macro_f1}"

        return False, "no_trigger_condition_met"

    def trigger_retraining(
        self,
        reason: str,
        sample_rows: int = 100_000,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes retraining request to the Training Microservice, verifies quality gate,
        and commands the Inference Microservice to reload the model if promoted.
        """
        in_cooldown, remaining = self.is_in_cooldown()
        if in_cooldown and not force:
            logger.info("Retraining blocked by cooldown (%d seconds remaining). Reason: %s", remaining, reason)
            return {
                "status": "blocked_by_cooldown",
                "seconds_remaining": remaining,
                "reason": reason,
            }

        logger.info("Initiating retraining run via %s/train. Reason: %s", self.training_url, reason)
        self.last_retrain_time = time.time()
        self.last_trigger_reason = reason

        reason_category = "manual"
        if "drift" in reason:
            reason_category = "drift"
        elif "performance" in reason:
            reason_category = "performance_degradation"
        RETRAINING_TRIGGERED_TOTAL.labels(reason=reason_category).inc()

        payload = {
            "sample_rows": sample_rows,
            "quality_gate_f1_threshold": self.min_macro_f1,
        }

        try:
            resp = requests.post(f"{self.training_url}/train", json=payload, timeout=300)
            if not resp.ok:
                err_detail = resp.text
                logger.error("Training service returned error %d: %s", resp.status_code, err_detail)
                self.last_run_result = {"status": "failed", "error": err_detail}
                return self.last_run_result

            train_data = resp.json()
            candidate_version = train_data.get("model_version")
            quality_gate = train_data.get("quality_gate", {})
            passed_gate = quality_gate.get("passed", False)

            reload_status = "not_requested"
            if passed_gate:
                # Model promoted -> instruct inference service to reload
                logger.info(
                    "Candidate model %s passed quality gate. Requesting inference reload...",
                    candidate_version,
                )
                try:
                    rel_resp = requests.post(f"{self.inference_url}/reload-model", timeout=10)
                    reload_status = "success" if rel_resp.ok else f"failed_{rel_resp.status_code}"
                except Exception as rel_err:
                    reload_status = f"failed_network: {rel_err}"
                    logger.warning("Could not reach inference service to reload: %s", rel_err)

            result = {
                "status": "success",
                "reason": reason,
                "model_version": candidate_version,
                "quality_gate": quality_gate,
                "model_reloaded": reload_status,
            }
            self.last_run_result = result
            return result

        except Exception as exc:
            logger.exception("Failed to trigger training run: %s", exc)
            self.last_run_result = {"status": "failed", "error": str(exc)}
            return self.last_run_result


# Global controller singleton
controller = RetrainingController()

