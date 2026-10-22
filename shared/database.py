"""
CryptoPulse - Shared Database Module
SQLAlchemy models, connection utilities, and persistence helpers for PostgreSQL.

Tables defined:
1. market_data: Real-time and historical 1-minute OHLCV candles.
2. predictions: Real-time 3-class prediction logs.
3. prediction_outcomes: Delayed ground-truth evaluations (t + 15m).
4. model_runs: Model training runs, macro F1, and promotion statuses.
5. drift_metrics: Feature drift statistics (PSI / KS) and drift alert statuses.
"""

import logging
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError

from shared.config import DATABASE_URL

logger = logging.getLogger("database")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

Base = declarative_base()

# Portable primary key: BigInteger for PostgreSQL (BIGSERIAL), Integer for SQLite (autoincrement)
ID_COLUMN = BigInteger().with_variant(Integer, "sqlite")


# =====================================================================
# Database Models
# =====================================================================

class MarketData(Base):
    """Stores 1-minute BTC/USDT candlestick market data."""
    __tablename__ = "market_data"

    id = Column(ID_COLUMN, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, default="BTCUSDT")
    candle_timestamp = Column(BigInteger, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(50), default="binance_rest", nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("symbol", "candle_timestamp", name="uq_symbol_candle_timestamp"),
        Index("idx_market_symbol_time", "symbol", "candle_timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.candle_timestamp,
            "candle_timestamp": self.candle_timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(Base):
    """Stores model inference outputs (UP, STABLE, DOWN) and class probabilities."""
    __tablename__ = "predictions"

    id = Column(ID_COLUMN, primary_key=True, autoincrement=True)
    prediction_timestamp = Column(BigInteger, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="BTCUSDT")
    predicted_class = Column(String(10), nullable=False)  # UP, STABLE, DOWN
    class_probabilities = Column(JSON, nullable=False)   # {"DOWN": 0.15, "STABLE": 0.70, "UP": 0.15}
    model_version = Column(String(50), nullable=False)
    evaluation_status = Column(String(20), default="PENDING", index=True, nullable=False)  # PENDING, EVALUATED
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    outcome = relationship("PredictionOutcome", back_populates="prediction", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "prediction_timestamp": self.prediction_timestamp,
            "symbol": self.symbol,
            "predicted_class": self.predicted_class,
            "class_probabilities": self.class_probabilities,
            "model_version": self.model_version,
            "evaluation_status": self.evaluation_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }
        return data


class PredictionOutcome(Base):
    """Stores delayed ground-truth evaluations once future candle prices arrive."""
    __tablename__ = "prediction_outcomes"

    id = Column(ID_COLUMN, primary_key=True, autoincrement=True)
    prediction_id = Column(ID_COLUMN, ForeignKey("predictions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    evaluation_timestamp = Column(BigInteger, nullable=False)
    future_close = Column(Float, nullable=False)
    actual_return = Column(Float, nullable=False)
    actual_class = Column(String(10), nullable=False)  # UP, STABLE, DOWN
    is_correct = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    prediction = relationship("Prediction", back_populates="outcome")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prediction_id": self.prediction_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "future_close": self.future_close,
            "actual_return": self.actual_return,
            "actual_class": self.actual_class,
            "is_correct": self.is_correct,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelRun(Base):
    """Tracks model training runs, macro F1, hyperparameters, and promotion status."""
    __tablename__ = "model_runs"

    id = Column(ID_COLUMN, primary_key=True, autoincrement=True)
    model_version = Column(String(50), unique=True, nullable=False)
    training_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    dataset_description = Column(String(255), nullable=True)
    accuracy = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    training_status = Column(String(20), default="COMPLETED", nullable=False)  # COMPLETED, FAILED
    promotion_status = Column(String(20), default="CANDIDATE", nullable=False) # CANDIDATE, PROMOTED, ARCHIVED
    mlflow_run_id = Column(String(100), nullable=True)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
            "dataset_description": self.dataset_description,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "training_status": self.training_status,
            "promotion_status": self.promotion_status,
            "mlflow_run_id": self.mlflow_run_id,
            "parameters": self.parameters,
            "metrics": self.metrics,
        }


class DriftMetric(Base):
    """Tracks continuous feature drift scores (PSI/KS) and drift alert states."""
    __tablename__ = "drift_metrics"

    id = Column(ID_COLUMN, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    drift_statistic = Column(Float, nullable=False)
    drift_status = Column(String(20), nullable=False)  # NO_DRIFT, DRIFT_DETECTED
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "feature_name": self.feature_name,
            "drift_statistic": self.drift_statistic,
            "drift_status": self.drift_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =====================================================================
# Database Engine & Session Management
# =====================================================================

def get_engine(db_url: Optional[str] = None):
    """
    Creates and returns a SQLAlchemy Engine.
    Handles connection pooling and SQLite dialect compatibility for local testing.
    """
    url = db_url or DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


# Default engine and sessionmaker
default_engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=default_engine)


def init_db(engine=None) -> None:
    """Creates all defined tables if they do not already exist."""
    target_engine = engine or default_engine
    Base.metadata.create_all(bind=target_engine)
    logger.info("Database tables initialized successfully on: %s", target_engine.url)


def get_db() -> Generator[Session, None, None]:
    """FastAPI and microservice dependency generator yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================================================
# Helper Persistence Functions
# =====================================================================

def save_market_candles(db: Session, candles: List[Dict[str, Any]]) -> int:
    """
    Persists parsed market candle dictionaries to the database.
    Skips duplicates based on the (symbol, candle_timestamp) unique constraint.

    Returns:
        int: Number of new candle records successfully inserted.
    """
    inserted_count = 0
    for c in candles:
        # Check if record already exists
        exists = (
            db.query(MarketData.id)
            .filter_by(symbol=c["symbol"], candle_timestamp=c["timestamp"])
            .first()
        )
        if exists:
            continue

        candle_record = MarketData(
            symbol=c["symbol"],
            candle_timestamp=c["timestamp"],
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            source=c.get("source", "binance_rest"),
        )
        db.add(candle_record)
        inserted_count += 1

    if inserted_count > 0:
        db.commit()
    return inserted_count


def record_prediction(db: Session, prediction_data: Dict[str, Any]) -> Prediction:
    """Saves a model prediction record."""
    pred = Prediction(
        prediction_timestamp=prediction_data["prediction_timestamp"],
        symbol=prediction_data.get("symbol", "BTCUSDT"),
        predicted_class=prediction_data["predicted_class"],
        class_probabilities=prediction_data["class_probabilities"],
        model_version=prediction_data["model_version"],
        evaluation_status="PENDING",
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def record_prediction_outcome(db: Session, outcome_data: Dict[str, Any]) -> PredictionOutcome:
    """
    Saves a delayed prediction evaluation outcome and updates the
    corresponding prediction's evaluation_status to 'EVALUATED'.
    """
    pred_id = outcome_data["prediction_id"]
    pred = db.query(Prediction).filter_by(id=pred_id).first()
    if pred:
        pred.evaluation_status = "EVALUATED"

    outcome = PredictionOutcome(
        prediction_id=pred_id,
        evaluation_timestamp=outcome_data["evaluation_timestamp"],
        future_close=outcome_data["future_close"],
        actual_return=outcome_data["actual_return"],
        actual_class=outcome_data["actual_class"],
        is_correct=outcome_data["is_correct"],
    )
    db.add(outcome)
    db.commit()
    db.refresh(outcome)
    return outcome


def record_model_run(db: Session, model_run_data: Dict[str, Any]) -> ModelRun:
    """Saves or updates a model training run record."""
    version = model_run_data["model_version"]
    run = db.query(ModelRun).filter_by(model_version=version).first()
    if not run:
        run = ModelRun(model_version=version)
        db.add(run)

    run.training_timestamp = model_run_data.get("training_timestamp", datetime.now(timezone.utc))
    run.dataset_description = model_run_data.get("dataset_description")
    run.accuracy = model_run_data.get("accuracy")
    run.macro_f1 = model_run_data.get("macro_f1")
    run.training_status = model_run_data.get("training_status", "COMPLETED")
    run.promotion_status = model_run_data.get("promotion_status", "CANDIDATE")
    run.mlflow_run_id = model_run_data.get("mlflow_run_id")
    run.parameters = model_run_data.get("parameters")
    run.metrics = model_run_data.get("metrics")

    db.commit()
    db.refresh(run)
    return run


def record_drift_metrics(db: Session, drift_records: List[Dict[str, Any]]) -> int:
    """Saves a batch of feature drift metrics."""
    count = 0
    for rec in drift_records:
        entry = DriftMetric(
            timestamp=rec["timestamp"],
            feature_name=rec["feature_name"],
            drift_statistic=rec["drift_statistic"],
            drift_status=rec["drift_status"],
        )
        db.add(entry)
        count += 1
    if count > 0:
        db.commit()
    return count
