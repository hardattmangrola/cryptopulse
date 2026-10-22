"""
Re-export BinanceClient and BinanceClientError from shared.binance_client
for backwards compatibility with existing imports.
"""

from shared.binance_client import BinanceClient, BinanceClientError

__all__ = ["BinanceClient", "BinanceClientError"]
