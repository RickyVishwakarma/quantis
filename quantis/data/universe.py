"""Instrument universe with sector tags.

Sectors drive the risk engine's sector-exposure limits. Symbols are NSE
tickers; the Yahoo Finance loader appends the ``.NS`` suffix.
"""

from __future__ import annotations

NIFTY50: dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "INFY": "IT",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "SBIN": "Financials",
    "BHARTIARTL": "Telecom",
    "KOTAKBANK": "Financials",
    "LT": "Industrials",
    "AXISBANK": "Financials",
    "ASIANPAINT": "Consumer",
    "MARUTI": "Auto",
    "SUNPHARMA": "Pharma",
    "TITAN": "Consumer",
    "BAJFINANCE": "Financials",
    "NESTLEIND": "FMCG",
    "WIPRO": "IT",
    "ULTRACEMCO": "Materials",
    "HCLTECH": "IT",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "TATAMOTORS": "Auto",
    "TATASTEEL": "Materials",
    "M&M": "Auto",
    "TECHM": "IT",
    "ADANIENT": "Conglomerate",
    "ADANIPORTS": "Infrastructure",
    "COALINDIA": "Energy",
    "BAJAJFINSV": "Financials",
    "ONGC": "Energy",
    "GRASIM": "Materials",
    "JSWSTEEL": "Materials",
    "HINDALCO": "Materials",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "EICHERMOT": "Auto",
    "BRITANNIA": "FMCG",
    "HEROMOTOCO": "Auto",
    "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare",
    "TATACONSUM": "FMCG",
    "BAJAJ-AUTO": "Auto",
    "INDUSINDBK": "Financials",
    "SBILIFE": "Financials",
    "HDFCLIFE": "Financials",
    "BPCL": "Energy",
    "SHRIRAMFIN": "Financials",
    "LTIM": "IT",
}


def symbols(universe: str = "NIFTY50", limit: int | None = None) -> list[str]:
    if universe.upper() != "NIFTY50":
        raise ValueError(f"Unknown universe: {universe!r} (MVP supports NIFTY50)")
    syms = list(NIFTY50)
    return syms[:limit] if limit else syms


def sector(symbol: str) -> str:
    return NIFTY50.get(symbol, "Unknown")
