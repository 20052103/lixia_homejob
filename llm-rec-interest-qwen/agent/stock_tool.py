# -*- coding: utf-8 -*-
"""
Stock market tool — near real-time quotes, news, and portfolio summaries.

Source priority:
  1. Finnhub  (realtime, free API key required)
  2. Polygon.io (15-min delayed on free plan, richer history)
  3. yfinance  (15-min delayed, no key — always available as last resort)
"""
from __future__ import annotations

import datetime as _dt
from typing import List, Optional

import requests

try:
    from .config import FINNHUB_API_KEY, POLYGON_API_KEY
except ImportError:
    from config import FINNHUB_API_KEY, POLYGON_API_KEY

_TIMEOUT = 8  # seconds per HTTP request


# ============================================================
# Internal helpers
# ============================================================

def _fmt_price(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)

def _fmt_pct(v) -> str:
    try:
        sign = "+" if float(v) >= 0 else ""
        return f"{sign}{float(v):.2f}%"
    except Exception:
        return str(v)

def _fmt_large(v) -> str:
    """Format large numbers (volume, market cap) compactly."""
    try:
        n = float(v)
        if n >= 1e12:
            return f"{n/1e12:.2f}T"
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(int(n))
    except Exception:
        return str(v)


# ============================================================
# Finnhub
# ============================================================

def _finnhub_quote(ticker: str) -> Optional[dict]:
    """Fetch real-time quote from Finnhub. Returns normalised dict or None."""
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker.upper(), "token": FINNHUB_API_KEY},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        # d keys: c=current, d=change, dp=change%, h=high, l=low, o=open, pc=prev_close, t=timestamp
        if not d.get("c"):
            return None
        return {
            "ticker": ticker.upper(),
            "price": d["c"],
            "change": d.get("d"),
            "change_pct": d.get("dp"),
            "high": d.get("h"),
            "low": d.get("l"),
            "open": d.get("o"),
            "prev_close": d.get("pc"),
            "source": "finnhub",
        }
    except Exception:
        return None


def _finnhub_news(ticker: str, n: int = 5) -> List[dict]:
    """Fetch recent company news from Finnhub."""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = _dt.date.today()
        week_ago = today - _dt.timedelta(days=7)
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker.upper(),
                "from": str(week_ago),
                "to": str(today),
                "token": FINNHUB_API_KEY,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json()[:n]
        return [
            {
                "headline": i.get("headline", ""),
                "source": i.get("source", ""),
                "date": _dt.datetime.fromtimestamp(i["datetime"]).strftime("%Y-%m-%d") if i.get("datetime") else "",
                "url": i.get("url", ""),
            }
            for i in items
        ]
    except Exception:
        return []


# ============================================================
# Polygon.io
# ============================================================

def _polygon_quote(ticker: str) -> Optional[dict]:
    """Fetch previous-day close from Polygon (free plan = prev day)."""
    if not POLYGON_API_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/prev",
            params={"adjusted": "true", "apiKey": POLYGON_API_KEY},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        d = r.json()
        results = d.get("results", [])
        if not results:
            return None
        res = results[0]
        price = res.get("c")  # closing price
        open_ = res.get("o")
        change = round(price - open_, 4) if price and open_ else None
        change_pct = round((change / open_) * 100, 2) if change and open_ else None
        return {
            "ticker": ticker.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "high": res.get("h"),
            "low": res.get("l"),
            "open": open_,
            "volume": res.get("v"),
            "source": "polygon (prev-day)",
        }
    except Exception:
        return None


def _polygon_news(ticker: str, n: int = 5) -> List[dict]:
    """Fetch recent ticker news from Polygon."""
    if not POLYGON_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker": ticker.upper(),
                "limit": n,
                "order": "desc",
                "apiKey": POLYGON_API_KEY,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        items = r.json().get("results", [])[:n]
        return [
            {
                "headline": i.get("title", ""),
                "source": (i.get("publisher") or {}).get("name", ""),
                "date": (i.get("published_utc") or "")[:10],
                "url": i.get("article_url", ""),
            }
            for i in items
        ]
    except Exception:
        return []


# ============================================================
# yfinance (fallback)
# ============================================================

def _yfinance_quote(ticker: str) -> Optional[dict]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return None
        prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change = round(price - prev, 4) if prev else None
        change_pct = round((change / prev) * 100, 2) if change and prev else None
        return {
            "ticker": ticker.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "open": info.get("open") or info.get("regularMarketOpen"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "source": "yfinance",
        }
    except Exception:
        return None


def _yfinance_news(ticker: str, n: int = 5) -> List[dict]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        news = t.news or []
        results = []
        for item in news[:n]:
            ts = item.get("providerPublishTime") or item.get("publishTime") or 0
            date = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
            # Handle both old and new yfinance news structure
            content = item.get("content") or {}
            headline = content.get("title") or item.get("title") or ""
            url = ""
            if content.get("canonicalUrl"):
                url = content["canonicalUrl"].get("url", "")
            elif item.get("link"):
                url = item["link"]
            results.append({
                "headline": headline,
                "source": item.get("publisher") or (content.get("provider") or {}).get("displayName", ""),
                "date": date,
                "url": url,
            })
        return results
    except Exception:
        return []


# ============================================================
# Public API
# ============================================================

def fetch_quote(ticker: str) -> dict:
    """
    Fetch a stock quote with Finnhub → Polygon → yfinance fallback.
    Returns a normalised dict; 'source' key indicates which was used.
    Raises ValueError if all sources fail.
    """
    ticker = ticker.strip().upper()
    for fn in (_finnhub_quote, _polygon_quote, _yfinance_quote):
        result = fn(ticker)
        if result:
            print(f"[stock] quote {ticker} via {result['source']}")
            return result
    raise ValueError(f"Could not fetch quote for {ticker!r} from any source.")


def fetch_news(ticker: str, n: int = 5) -> List[dict]:
    """
    Fetch recent news headlines. Tries Finnhub → Polygon → yfinance.
    Returns a (possibly empty) list.
    """
    ticker = ticker.strip().upper()
    for fn in (
        lambda t: _finnhub_news(t, n),
        lambda t: _polygon_news(t, n),
        lambda t: _yfinance_news(t, n),
    ):
        items = fn(ticker)
        if items:
            return items
    return []


def format_quote(q: dict, include_news: bool = True) -> str:
    """Render a quote dict as a compact human-readable string for the LLM."""
    ticker = q.get("ticker", "?")
    price = _fmt_price(q.get("price"))
    chg = _fmt_pct(q.get("change_pct")) if q.get("change_pct") is not None else "N/A"
    high = _fmt_price(q["high"]) if q.get("high") else "N/A"
    low = _fmt_price(q["low"]) if q.get("low") else "N/A"
    vol = _fmt_large(q["volume"]) if q.get("volume") else "N/A"
    cap = _fmt_large(q["market_cap"]) if q.get("market_cap") else "N/A"
    pe = f"{float(q['pe_ratio']):.1f}" if q.get("pe_ratio") else "N/A"
    w52h = _fmt_price(q["week_52_high"]) if q.get("week_52_high") else "N/A"
    w52l = _fmt_price(q["week_52_low"]) if q.get("week_52_low") else "N/A"
    source = q.get("source", "?")

    lines = [
        f"📈 {ticker}  {price}  {chg}",
        f"   Day: {_fmt_price(q['low']) if q.get('low') else 'N/A'} – {_fmt_price(q['high']) if q.get('high') else 'N/A'}  |  Vol: {vol}",
    ]
    if q.get("market_cap"):
        lines.append(f"   Market Cap: {cap}  |  P/E: {pe}")
    if q.get("week_52_high"):
        lines.append(f"   52w: {w52l} – {w52h}")
    lines.append(f"   [source: {source}]")

    if include_news:
        news = fetch_news(ticker)
        if news:
            lines.append("\n📰 Recent News:")
            for item in news:
                date = f"[{item['date']}] " if item.get("date") else ""
                src = f" — {item['source']}" if item.get("source") else ""
                lines.append(f"  • {date}{item['headline']}{src}")

    return "\n".join(lines)


def fetch_stock_summary(tickers: List[str], include_news: bool = True) -> str:
    """
    Fetch quotes (and optionally news) for one or more tickers.
    Returns a formatted multi-stock summary string.
    """
    if not tickers:
        return "No tickers provided."

    parts = []
    for ticker in tickers:
        try:
            q = fetch_quote(ticker)
            parts.append(format_quote(q, include_news=include_news and len(tickers) == 1))
        except ValueError as e:
            parts.append(f"⚠️  {ticker}: {e}")

    # For multi-stock, append news for each after prices
    if include_news and len(tickers) > 1:
        parts.append("\n📰 Recent News:")
        for ticker in tickers:
            news = fetch_news(ticker, n=3)
            if news:
                parts.append(f"\n  {ticker}:")
                for item in news:
                    date = f"[{item['date']}] " if item.get("date") else ""
                    parts.append(f"    • {date}{item['headline']}")

    return "\n\n".join(parts)
