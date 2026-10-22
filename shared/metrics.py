"""
CryptoPulse - Shared Prometheus Metrics Module
Standard Prometheus metrics exposed across microservices via GET /metrics.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# -------------------------------------------------------------
# Common HTTP Service Metrics
# -------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "cryptopulse_http_requests_total",
    "Total HTTP requests received",
    ["service", "method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cryptopulse_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["service", "endpoint"]
)

# -------------------------------------------------------------
# Ingestion Service Metrics
# -------------------------------------------------------------
BINANCE_FETCH_TOTAL = Counter(
    "cryptopulse_binance_fetch_total",
    "Total Binance market data API poll attempts",
    ["status"]  # 'success', 'failure'
)

CANDLES_INGESTED_TOTAL = Counter(
    "cryptopulse_candles_ingested_total",
    "Total completed 1-minute candlestick records saved to database"
)

# -------------------------------------------------------------
# Inference Service Metrics
# -------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "cryptopulse_predictions_total",
    "Total inference predictions generated",
    ["predicted_class"]  # 'UP', 'STABLE', 'DOWN'
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "cryptopulse_prediction_latency_seconds",
    "Time taken to engineer features and run model inference"
)

MODEL_VERSION_GAUGE = Gauge(
    "cryptopulse_active_model_version",
    "Active model version loaded in inference service",
    ["version"]
)

# -------------------------------------------------------------
# Training Service Metrics
# -------------------------------------------------------------
TRAINING_RUNS_TOTAL = Counter(
    "cryptopulse_training_runs_total",
    "Total model training jobs triggered",
    ["status"]  # 'success', 'failed'
)

MODEL_TEST_MACRO_F1 = Gauge(
    "cryptopulse_model_test_macro_f1",
    "Macro F1 score of the currently approved model"
)


# -------------------------------------------------------------
# Continuous Monitoring & Maintenance Metrics (Phase 7)
# -------------------------------------------------------------
FEATURE_DRIFT_PSI = Gauge(
    "cryptopulse_feature_drift_psi",
    "Population Stability Index (PSI) per engineered feature",
    ["feature"]
)

DRIFT_ALERTS_TOTAL = Counter(
    "cryptopulse_drift_alerts_total",
    "Total feature drift threshold breaches (PSI >= 0.25)"
)

MODEL_ACCURACY_ROLLING = Gauge(
    "cryptopulse_model_rolling_accuracy",
    "Rolling accuracy evaluated against 1-minute ground-truth outcomes"
)

MODEL_MACRO_F1_ROLLING = Gauge(
    "cryptopulse_model_rolling_macro_f1",
    "Rolling Macro F1 evaluated against 1-minute ground-truth outcomes"
)

PREDICTION_OUTCOMES_TOTAL = Counter(
    "cryptopulse_prediction_outcomes_total",
    "Total delayed 1-minute ground truth prediction evaluations",
    ["actual_class", "is_correct"]
)

RETRAINING_TRIGGERED_TOTAL = Counter(
    "cryptopulse_retraining_triggered_total",
    "Total automated retraining runs triggered by controller",
    ["reason"]  # 'drift', 'performance_degradation', 'manual'
)


def get_metrics_response() -> Response:
    """Helper returning Prometheus plain-text metrics response for FastAPI."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

