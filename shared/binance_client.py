"""
CryptoPulse - Shared Binance Public Market Data REST Client
Provides market candle fetching and OHLCV parsing shared across microservices.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import requests

from shared.config import (
    BINANCE_API_BASE_URL,
    BINANCE_SYMBOL,
    BINANCE_CANDLE_INTERVAL,
    BINANCE_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger("binance_client")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


class BinanceClientError(Exception):
    """Custom exception raised when Binance API request fails or returns invalid data."""
    pass


class BinanceClient:
    """Client for fetching public Bitcoin candlestick data from Binance REST API."""

    def __init__(
        self,
        base_url: str = BINANCE_API_BASE_URL,
        symbol: str = BINANCE_SYMBOL,
        candle_interval: str = BINANCE_CANDLE_INTERVAL,
        timeout_seconds: int = BINANCE_REQUEST_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol.upper()
        self.candle_interval = candle_interval
        self.timeout_seconds = timeout_seconds

        # Configure session with standard user agent header
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CryptoPulse-MLOps/1.0"})

    def parse_kline(self, raw_kline: list) -> Dict[str, Any]:
        """
        Parses a single raw Binance kline array into a standardized candle dictionary.
        """
        if not isinstance(raw_kline, (list, tuple)) or len(raw_kline) < 7:
            raise ValueError(f"Invalid raw kline structure: {raw_kline}")

        open_time_ms = int(raw_kline[0])
        close_time_ms = int(raw_kline[6])

        # Current epoch time in milliseconds
        now_ms = int(time.time() * 1000)

        # Candle is complete if current time has passed the candle close time
        is_completed = now_ms >= close_time_ms

        return {
            "timestamp": open_time_ms // 1000,         # Standard epoch in seconds (matches CSV)
            "open_time_ms": open_time_ms,
            "close_time_ms": close_time_ms,
            "symbol": self.symbol,
            "open": float(raw_kline[1]),
            "high": float(raw_kline[2]),
            "low": float(raw_kline[3]),
            "close": float(raw_kline[4]),
            "volume": float(raw_kline[5]),
            "is_completed": is_completed,
            "source": "binance_rest"
        }

    def fetch_klines(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        limit: int = 10,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches candlestick records from Binance API endpoint /api/v3/klines.
        """
        target_symbol = (symbol or self.symbol).upper()
        target_interval = interval or self.candle_interval

        endpoint = f"{self.base_url}/api/v3/klines"
        params: Dict[str, Any] = {
            "symbol": target_symbol,
            "interval": target_interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout_seconds)

            if response.status_code != 200:
                error_msg = (
                    f"Binance API returned error status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                logger.error(error_msg)
                raise BinanceClientError(error_msg)

            raw_candles = response.json()
            if not isinstance(raw_candles, list):
                error_msg = f"Unexpected response format from Binance API: {raw_candles}"
                logger.error(error_msg)
                raise BinanceClientError(error_msg)

            parsed_candles = [self.parse_kline(item) for item in raw_candles]
            return parsed_candles

        except requests.exceptions.Timeout as exc:
            error_msg = f"Timeout connecting to Binance API after {self.timeout_seconds}s"
            logger.error(error_msg)
            raise BinanceClientError(error_msg) from exc

        except requests.exceptions.RequestException as exc:
            error_msg = f"Network connection error contacting Binance API: {exc}"
            logger.error(error_msg)
            raise BinanceClientError(error_msg) from exc

    def fetch_completed_candles(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieves only fully completed candles."""
        all_candles = self.fetch_klines(symbol=symbol, interval=interval, limit=limit)
        completed = [c for c in all_candles if c["is_completed"]]
        return completed

    @staticmethod
    def deduplicate_candles(
        new_candles: List[Dict[str, Any]],
        existing_timestamps: set[int]
    ) -> List[Dict[str, Any]]:
        """Filters out candles whose timestamp has already been recorded."""
        unique_candles = []
        for candle in new_candles:
            ts = candle["timestamp"]
            if ts not in existing_timestamps:
                unique_candles.append(candle)
                existing_timestamps.add(ts)
        return unique_candles
