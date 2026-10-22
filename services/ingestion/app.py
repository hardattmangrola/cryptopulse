"""
CryptoPulse - Ingestion Microservice
Continuously polls Binance public REST API for completed 1-minute BTC/USDT candles
and persists them into PostgreSQL while preventing duplicates.
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

from fastapi import FastAPI, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import Response

from shared.config import (
    BINANCE_SYMBOL,
    BINANCE_CANDLE_INTERVAL,
    BINANCE_POLL_INTERVAL_SECONDS,
)
from shared.database import get_db, init_db, save_market_candles, MarketData
from shared.metrics import (
    BINANCE_FETCH_TOTAL,
    CANDLES_INGESTED_TOTAL,
    HTTP_REQUESTS_TOTAL,
    get_metrics_response,
)
from services.ingestion.binance_client import BinanceClient, BinanceClientError

logger = logging.getLogger("ingestion_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global ingestion state
ingestion_state: Dict[str, Any] = {
    "status": "starting",
    "last_poll_time": None,
    "last_poll_status": "none",
    "last_error": None,
    "total_ingested": 0,
    "is_polling_active": True,
}

binance_client = BinanceClient()


def run_single_poll(db: Session) -> Dict[str, Any]:
    """
    Executes a single market data fetch from Binance and persists completed candles to the database.
    """
    try:
        # Fetch completed candles
        completed_candles = binance_client.fetch_completed_candles(limit=15)
        BINANCE_FETCH_TOTAL.labels(status="success").inc()

        # Persist to database (deduplicating against existing timestamps)
        inserted_count = save_market_candles(db, completed_candles)
        CANDLES_INGESTED_TOTAL.inc(inserted_count)

        ingestion_state["last_poll_time"] = int(time.time())
        ingestion_state["last_poll_status"] = "success"
        ingestion_state["last_error"] = None
        ingestion_state["total_ingested"] += inserted_count

        logger.info(
            "Polled Binance (%s %s): %d completed candles retrieved, %d new records saved.",
            BINANCE_SYMBOL,
            BINANCE_CANDLE_INTERVAL,
            len(completed_candles),
            inserted_count,
        )

        return {
            "status": "success",
            "candles_fetched": len(completed_candles),
            "new_candles_saved": inserted_count,
            "latest_candle_time": completed_candles[-1]["timestamp"] if completed_candles else None,
        }

    except BinanceClientError as exc:
        BINANCE_FETCH_TOTAL.labels(status="failure").inc()
        ingestion_state["last_poll_time"] = int(time.time())
        ingestion_state["last_poll_status"] = "failure"
        ingestion_state["last_error"] = str(exc)
        logger.error("Binance ingestion error (service continuing safely): %s", exc)
        return {"status": "error", "error": str(exc)}

    except Exception as exc:
        BINANCE_FETCH_TOTAL.labels(status="failure").inc()
        ingestion_state["last_poll_time"] = int(time.time())
        ingestion_state["last_poll_status"] = "failure"
        ingestion_state["last_error"] = str(exc)
        logger.exception("Unexpected error during ingestion poll: %s", exc)
        return {"status": "error", "error": str(exc)}


async def background_poll_loop():
    """Background asynchronous task that executes the periodic Binance polling."""
    logger.info("Starting background Binance polling loop (interval=%ds)...", BINANCE_POLL_INTERVAL_SECONDS)
    # Ensure tables exist
    try:
        init_db()
    except Exception as e:
        logger.warning("Could not initialize DB on startup (will retry on poll): %s", e)

    while ingestion_state["is_polling_active"]:
        # Run poll in database session
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                run_single_poll(db)
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception as e:
            logger.warning("Database unavailable during poll loop: %s", e)

        await asyncio.sleep(BINANCE_POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan managing the background polling task."""
    ingestion_state["status"] = "running"
    task = asyncio.create_task(background_poll_loop())
    yield
    ingestion_state["is_polling_active"] = False
    ingestion_state["status"] = "stopped"
    task.cancel()


app = FastAPI(
    title="CryptoPulse - Data Ingestion Microservice",
    description="Periodically collects completed 1-minute Bitcoin candlestick data from Binance API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health & Monitoring"])
def health_check():
    """Returns service health status, last poll result, and ingestion statistics."""
    HTTP_REQUESTS_TOTAL.labels(service="ingestion", method="GET", endpoint="/health", status="200").inc()
    return {
        "service": "ingestion",
        "status": "healthy" if ingestion_state["status"] == "running" else ingestion_state["status"],
        "symbol": BINANCE_SYMBOL,
        "interval": BINANCE_CANDLE_INTERVAL,
        "poll_interval_seconds": BINANCE_POLL_INTERVAL_SECONDS,
        "last_poll_time": ingestion_state["last_poll_time"],
        "last_poll_status": ingestion_state["last_poll_status"],
        "last_error": ingestion_state["last_error"],
        "total_candles_ingested": ingestion_state["total_ingested"],
    }


@app.get("/metrics", tags=["Health & Monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return get_metrics_response()


@app.post("/poll", tags=["Ingestion Control"])
def trigger_manual_poll(db: Session = Depends(get_db)):
    """Manually triggers an immediate Binance market data fetch and database save."""
    HTTP_REQUESTS_TOTAL.labels(service="ingestion", method="POST", endpoint="/poll", status="200").inc()
    result = run_single_poll(db)
    return result


@app.get("/candles/recent", tags=["Market Data"])
def get_recent_candles(
    limit: int = Query(default=20, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Returns the most recent completed market candles stored in the database."""
    HTTP_REQUESTS_TOTAL.labels(service="ingestion", method="GET", endpoint="/candles/recent", status="200").inc()
    records = (
        db.query(MarketData)
        .filter_by(symbol=BINANCE_SYMBOL)
        .order_by(MarketData.candle_timestamp.desc())
        .limit(limit)
        .all()
    )
    # Return chronologically ascending
    return [r.to_dict() for r in reversed(records)]
