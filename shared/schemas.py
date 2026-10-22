"""
CryptoPulse - Shared Pydantic Schemas
Defines request, response, and database transfer schemas used across microservices.
"""

from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field, ConfigDict


# =====================================================================
# Market Data Schemas
# =====================================================================

class MarketCandleBase(BaseModel):
    symbol: str = "BTCUSDT"
    timestamp: int = Field(..., description="Candle start unix timestamp in seconds")
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "binance_rest"


class MarketCandleCreate(MarketCandleBase):
    is_completed: bool = True


class MarketCandleRead(MarketCandleBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Inference & Prediction Schemas
# =====================================================================

class PredictionProbabilities(BaseModel):
    DOWN: float = Field(..., ge=0.0, le=1.0)
    STABLE: float = Field(..., ge=0.0, le=1.0)
    UP: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """
    Standard format matching project specification:
    {
      "symbol": "BTCUSDT",
      "prediction": "UP",
      "probabilities": { "DOWN": 0.15, "STABLE": 0.20, "UP": 0.65 },
      "model_version": "1.0.0",
      "prediction_timestamp": 1785258240,
      "prediction_id": 1
    }
    """
    symbol: str
    prediction: str
    probabilities: PredictionProbabilities
    model_version: str
    prediction_timestamp: int
    prediction_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionRead(BaseModel):
    id: int
    prediction_timestamp: int
    symbol: str
    predicted_class: str
    class_probabilities: Dict[str, float]
    model_version: str
    evaluation_status: str

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Delayed Evaluation Schemas
# =====================================================================

class PredictionOutcomeRead(BaseModel):
    id: int
    prediction_id: int
    evaluation_timestamp: int
    future_close: float
    actual_return: float
    actual_class: str
    is_correct: bool

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Model Run & Metadata Schemas
# =====================================================================

class ModelRunRead(BaseModel):
    id: int
    model_version: str
    training_timestamp: Any
    dataset_description: Optional[str] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    training_status: str
    promotion_status: str
    mlflow_run_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Drift Monitoring Schemas
# =====================================================================

class DriftMetricRead(BaseModel):
    id: int
    timestamp: int
    feature_name: str
    drift_statistic: float
    drift_status: str

    model_config = ConfigDict(from_attributes=True)
