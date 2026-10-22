"""
CryptoPulse - Configuration Module
Centralizes application settings loaded from environment variables or sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def _resolve_historical_data_path() -> Path:
    env_path = os.getenv("HISTORICAL_DATA_PATH")
    candidates = []
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else (BASE_DIR / p).resolve())
    candidates.extend([
        BASE_DIR / "btcusd_1-min_data.csv",
        Path("/app/btcusd_1-min_data.csv"),
        BASE_DIR / "app" / "btcusd_1-min_data.csv",
        Path("btcusd_1-min_data.csv").resolve(),
    ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return candidates[0]

# Path to the historical Bitcoin 1-minute dataset
HISTORICAL_DATA_PATH = _resolve_historical_data_path()

# Expected CSV schema based on the provided btcusd_1-min_data.csv
EXPECTED_CSV_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]

# Binance Public Market Data API settings
BINANCE_API_BASE_URL = os.getenv("BINANCE_API_BASE_URL", "https://api.binance.com").rstrip("/")
BINANCE_SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
BINANCE_CANDLE_INTERVAL = os.getenv("BINANCE_CANDLE_INTERVAL", "1m")
BINANCE_POLL_INTERVAL_SECONDS = int(os.getenv("BINANCE_POLL_INTERVAL_SECONDS", "60"))
BINANCE_REQUEST_TIMEOUT_SECONDS = int(os.getenv("BINANCE_REQUEST_TIMEOUT_SECONDS", "10"))

# PostgreSQL Database Configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cryptopulse")

DEFAULT_PG_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Docker environment detection
_IS_DOCKER = Path("/.dockerenv").exists() or os.getenv("DOCKER_CONTAINER") == "true"

# MLflow Experiment Tracking Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000" if _IS_DOCKER else "http://localhost:5000"
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "cryptopulse_btc_movement")

# Phase 7: Continuous Maintenance & 1-Minute Horizon Configuration
PREDICTION_HORIZON_SECONDS = int(os.getenv("PREDICTION_HORIZON_SECONDS", "60"))
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.0005"))  # 0.05% return in 1 minute
EVALUATION_POLL_INTERVAL_SECONDS = int(os.getenv("EVALUATION_POLL_INTERVAL_SECONDS", "5"))
DRIFT_CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "60"))
PSI_DRIFT_THRESHOLD = float(os.getenv("PSI_DRIFT_THRESHOLD", "0.25"))
RETRAINING_COOLDOWN_MINUTES = int(os.getenv("RETRAINING_COOLDOWN_MINUTES", "10"))
RETRAINING_MIN_MACRO_F1 = float(os.getenv("RETRAINING_MIN_MACRO_F1", "0.30"))

# Service Inter-communication URLs
TRAINING_SERVICE_URL = os.getenv(
    "TRAINING_SERVICE_URL",
    "http://training:8003" if _IS_DOCKER else "http://localhost:8003"
)
INFERENCE_SERVICE_URL = os.getenv(
    "INFERENCE_SERVICE_URL",
    "http://inference:8002" if _IS_DOCKER else "http://localhost:8002"
)


