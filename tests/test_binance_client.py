"""
Unit tests for services/ingestion/binance_client.py
Tests candle parsing, timestamp conversions, completed candle logic,
deduplication, and error handling with mocked API responses.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
import requests
from services.ingestion.binance_client import BinanceClient, BinanceClientError


# Sample raw Binance Kline response matching official documentation:
# [open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_base, taker_quote, ignore]
SAMPLE_RAW_KLINE = [
    1700000000000,          # Open time ms (equals 1700000000 seconds)
    "65000.50",             # Open
    "65100.00",             # High
    "64950.25",             # Low
    "65080.00",             # Close
    "12.456",               # Volume
    1700000059999,          # Close time ms
    "810620.00",            # Quote asset volume
    350,                    # Number of trades
    "6.120",                # Taker buy base asset volume
    "398000.00",            # Taker buy quote asset volume
    "0"                     # Ignore
]


def test_parse_kline_structure():
    """Verify raw Binance array is properly converted into typed dictionary."""
    client = BinanceClient()
    parsed = client.parse_kline(SAMPLE_RAW_KLINE)

    assert parsed["timestamp"] == 1700000000
    assert parsed["open_time_ms"] == 1700000000000
    assert parsed["close_time_ms"] == 1700000059999
    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["open"] == 65000.50
    assert parsed["high"] == 65100.00
    assert parsed["low"] == 64950.25
    assert parsed["close"] == 65080.00
    assert parsed["volume"] == 12.456
    assert parsed["source"] == "binance_rest"
    # Since 1700000059999 ms is in the past, is_completed must be True
    assert parsed["is_completed"] is True


def test_parse_kline_in_progress():
    """Verify that a candle whose close_time is in the future is marked as in-progress."""
    client = BinanceClient()
    future_close_ms = int(time.time() * 1000) + 60000  # 1 minute into future
    kline_in_progress = list(SAMPLE_RAW_KLINE)
    kline_in_progress[6] = future_close_ms

    parsed = client.parse_kline(kline_in_progress)
    assert parsed["is_completed"] is False


def test_parse_kline_invalid_data():
    """Verify ValueError is raised if raw kline does not have enough elements."""
    client = BinanceClient()
    with pytest.raises(ValueError):
        client.parse_kline([1700000000000, "65000.0"])


@patch.object(requests.Session, "get")
def test_fetch_klines_success(mock_get):
    """Verify successful retrieval and parsing of candles when API returns 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [SAMPLE_RAW_KLINE]
    mock_get.return_value = mock_resp

    client = BinanceClient()
    candles = client.fetch_klines(limit=1)

    assert len(candles) == 1
    assert candles[0]["symbol"] == "BTCUSDT"
    assert candles[0]["close"] == 65080.00


@patch.object(requests.Session, "get")
def test_fetch_klines_http_error(mock_get):
    """Verify BinanceClientError is raised on HTTP 500 error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_get.return_value = mock_resp

    client = BinanceClient()
    with pytest.raises(BinanceClientError) as exc_info:
        client.fetch_klines(limit=1)
    assert "returned error status 500" in str(exc_info.value)


@patch.object(requests.Session, "get")
def test_fetch_klines_timeout(mock_get):
    """Verify BinanceClientError is raised on request timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

    client = BinanceClient()
    with pytest.raises(BinanceClientError) as exc_info:
        client.fetch_klines(limit=1)
    assert "Timeout connecting to Binance" in str(exc_info.value)


def test_deduplicate_candles():
    """Verify deduplication removes candles whose timestamps are already recorded."""
    candles = [
        {"timestamp": 100, "close": 50000.0},
        {"timestamp": 101, "close": 50010.0},
        {"timestamp": 102, "close": 50020.0},
    ]
    seen_timestamps = {100}

    filtered = BinanceClient.deduplicate_candles(candles, seen_timestamps)
    assert len(filtered) == 2
    assert [c["timestamp"] for c in filtered] == [101, 102]
    # Check that seen_timestamps set has been updated
    assert seen_timestamps == {100, 101, 102}
