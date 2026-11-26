"""
TradeSense AI — Single-File Python Web App
Run: python trade_bot.py
Then open: http://localhost:8000

Requirements:
    pip install fastapi uvicorn pandas numpy groq requests nsepython
"""

import os, sys, json, asyncio
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List

import time, random
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    print(f"✅ Groq API key loaded (ends with ...{GROQ_API_KEY[-6:]})")
else:
    print("⚠️  Warning: GROQ_API_KEY not set — chatbot will not work")

if POLYGON_API_KEY and POLYGON_API_KEY != "your_polygon_api_key_here":
    print(f"✅ Polygon API key loaded (ends with ...{POLYGON_API_KEY[-6:]})")
else:
    print("⚠️  Warning: POLYGON_API_KEY not set — stock charts will not work")

# ── Polygon.io data helpers ────────────────────────────────────────────────────
POLYGON_BASE = "https://api.polygon.io"
_CACHE: dict = {}
_CACHE_TTL   = 300  # 5-min cache

_PERIOD_MAP = {
    "1d":  (1,   1, "minute"),
    "5d":  (5,   5, "minute"),
    "1mo": (30,  1, "day"),
    "3mo": (90,  1, "day"),
    "6mo": (180, 1, "day"),
    "1y":  (365, 1, "day"),
    "2y":  (730, 1, "week"),
    "5y":  (1825,1, "week"),
}

def _poly(path: str, params: dict = {}) -> dict:
    cache_key = path + str(sorted(params.items()))
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data
    r = requests.get(f"{POLYGON_BASE}{path}",
                     params={**params, "apiKey": POLYGON_API_KEY}, timeout=15)
    if r.status_code == 429:
        raise Exception("Rate limit hit — please wait a moment and try again.")
    if r.status_code == 403:
        raise Exception("Invalid Polygon API key. Check the POLYGON_API_KEY environment variable.")
    r.raise_for_status()
    data = r.json()
    _CACHE[cache_key] = (now, data)
    return data

def _get_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    days, mult, span = _PERIOD_MAP.get(period, (90, 1, "day"))
    if interval == "1h":   mult, span = 1, "hour"
    elif interval == "1wk": mult, span = 1, "week"
    elif interval == "1mo": mult, span = 1, "month"

    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    to_date   = pd.Timestamp.now().strftime("%Y-%m-%d")
    url       = f"/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{from_date}/{to_date}"
    data      = _poly(url, {"adjusted": "true", "sort": "asc", "limit": 50000})

    if data.get("status") == "ERROR" or not data.get("results"):
        raise Exception(f"No data for '{symbol}'. Check the ticker symbol (use US format e.g. AAPL).")

    rows = [{"date": pd.Timestamp(b["t"], unit="ms").strftime("%Y-%m-%d"),
             "open": b["o"], "high": b["h"], "low": b["l"],
             "close": b["c"], "volume": b.get("v", 0)}
            for b in data["results"]]
    return pd.DataFrame(rows)

def _get_profile(symbol: str) -> dict:
    try:
        r = _poly(f"/v3/reference/tickers/{symbol}").get("results", {})
        return {"name": r.get("name", symbol),
                "sector": r.get("sic_description", ""),
                "market_cap": r.get("market_cap")}
    except:
        return {"name": symbol, "sector": "", "market_cap": None}

def _get_metrics(symbol: str) -> dict:
    try:
        results = _poly("/vX/reference/financials",
                        {"ticker": symbol, "limit": 1}).get("results", [])
        if not results: return {}
        inc = results[0].get("financials", {}).get("income_statement", {})
        return {"eps": inc.get("basic_earnings_per_share", {}).get("value")}
    except:
        return {}


# ── NSEPython fallback (Indian .NS / .BO stocks + any Polygon miss) ───────────
from nsepython import nse_eq, nsefetch

def _is_indian_symbol(symbol: str) -> bool:
    """Detect Indian exchange tickers — .NS (NSE) or .BO (BSE)."""
    return symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO")

def _nse_base(symbol: str) -> str:
    """Strip exchange suffix to get bare NSE ticker (e.g. SBIN.NS → SBIN)."""
    return symbol.upper().replace(".NS", "").replace(".BO", "")

def _yf_fetch(yf_symbol: str, yf_period: str, yf_interval: str):
    """Try multiple yfinance strategies to get around the broken Indian stock issue."""
    import yfinance as yf

    # ── Strategy 1: standard .history() ──────────────────────────────────────
    try:
        t = yf.Ticker(yf_symbol)
        h = t.history(period=yf_period, interval=yf_interval, auto_adjust=True, repair=True)
        if not h.empty:
            return h
    except Exception:
        pass

    # ── Strategy 2: yf.download() — uses a different internal code path ──────
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):   # silence noisy warnings
            h = yf.download(
                yf_symbol, period=yf_period, interval=yf_interval,
                auto_adjust=True, progress=False, threads=False,
            )
        if not h.empty:
            # download() returns MultiIndex columns when single ticker — flatten
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            return h
    except Exception:
        pass

    # ── Strategy 3: requests + Yahoo Finance v8 JSON directly ────────────────
    try:
        import requests as _req, time as _time
        _RANGE_MAP = {
            "1d":"1d","5d":"5d","1mo":"1mo","3mo":"3mo",
            "6mo":"6mo","1y":"1y","2y":"2y","5y":"5y",
        }
        _INT_MAP = {
            "5m":"5m","30m":"30m","1h":"1h","1d":"1d","1wk":"1wk","1mo":"1mo",
        }
        url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
               f"?range={yf_period}&interval={yf_interval}&includePrePost=false")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = _req.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts   = res["timestamp"]
        q    = res["indicators"]["quote"][0]
        adj  = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
        rows = []
        for i, t_ in enumerate(ts):
            if q["close"][i] is None: continue
            rows.append({
                "Date":   pd.Timestamp(t_, unit="s", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None),
                "Open":   q["open"][i],  "High":  q["high"][i],
                "Low":    q["low"][i],   "Close": adj[i],
                "Volume": q["volume"][i] or 0,
            })
        h = pd.DataFrame(rows).set_index("Date")
        if not h.empty:
            return h
    except Exception:
        pass

    return pd.DataFrame()   # all strategies failed


def _nse_get_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history for Indian stocks — 3-layer fallback for yfinance reliability."""

    _YF_PERIOD = {
        "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo",
        "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y",
    }
    yf_period   = _YF_PERIOD.get(period, "3mo")
    yf_interval = "1d"                   # default
    if period == "1d":   yf_interval = "5m"
    elif period == "5d": yf_interval = "30m"
    elif interval in ("1wk", "1mo"): yf_interval = interval

    # Ensure .NS suffix; also prepare .BO fallback
    yf_symbol = symbol.upper()
    if not (yf_symbol.endswith(".NS") or yf_symbol.endswith(".BO")):
        yf_symbol += ".NS"
    alt_symbol = yf_symbol.replace(".NS", ".BO") if yf_symbol.endswith(".NS") else yf_symbol.replace(".BO", ".NS")

    hist = _yf_fetch(yf_symbol, yf_period, yf_interval)
    if hist.empty:
        print(f"  [{symbol}] Primary fetch empty — trying {alt_symbol}")
        hist = _yf_fetch(alt_symbol, yf_period, yf_interval)

    if hist.empty:
        raise Exception(
            f"No data found for '{symbol}'. "
            "Tips: use SBIN.NS, RELIANCE.NS, TCS.NS — NSE tickers with .NS suffix. "
            "Run: pip install --upgrade yfinance"
        )

    hist = hist.reset_index()
    # Normalise the date column (different strategies return different column names)
    date_col = next((c for c in hist.columns if c in ("Datetime", "Date", "index")), hist.columns[0])
    hist["date"] = pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d")

    # Normalise OHLCV column names (yf.download capitalises, direct JSON may not)
    col_map = {}
    for c in hist.columns:
        cl = c.lower()
        if cl in ("open","high","low","close","volume","adj close","adjclose"):
            col_map[c] = cl.replace(" ","").replace("adj","").strip() if "adj" not in cl else "close"
    hist = hist.rename(columns=col_map)

    needed = {"open","high","low","close","volume"}
    missing = needed - set(hist.columns)
    if missing:
        # last-ditch: try to rename whatever numeric columns exist
        num_cols = hist.select_dtypes(include="number").columns.tolist()
        for nc, name in zip(num_cols, ["open","high","low","close","volume"]):
            hist.rename(columns={nc: name}, inplace=True)

    df = hist[["date","open","high","low","close","volume"]].copy()
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].apply(pd.to_numeric, errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df

def _nse_get_profile(symbol: str) -> dict:
    try:
        base = _nse_base(symbol)
        q    = nse_eq(base)
        info = q.get("info", {})
        meta = q.get("metadata", {})
        mktcap = None
        try:
            shares = float(meta.get("pdSectorIndAll", "").split("|")[0] or 0)
            ltp    = float(q.get("priceInfo", {}).get("lastPrice", 0))
            if shares and ltp:
                mktcap = shares * ltp
        except: pass
        return {
            "name":       info.get("companyName") or base,
            "sector":     meta.get("industry", ""),
            "market_cap": mktcap,
        }
    except:
        return {"name": symbol, "sector": "", "market_cap": None}

def _nse_get_metrics(symbol: str) -> dict:
    try:
        base = _nse_base(symbol)
        q    = nse_eq(base)
        pd_  = q.get("priceInfo", {})
        eps, pe = None, None
        # NSE provides P/E directly in some responses
        try: pe = float(q.get("metadata", {}).get("pdSymbolPe") or 0) or None
        except: pass
        return {"eps": eps, "pe_ratio": pe}
    except:
        return {}

# ── Unified wrappers (NSEPython for Indian, Polygon for US) ───────────────────
def get_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if _is_indian_symbol(symbol):
        return _nse_get_candles(symbol, period, interval)
    try:
        return _get_candles(symbol, period, interval)
    except Exception:
        # Last-resort: try NSE even for unknown symbols
        return _nse_get_candles(symbol, period, interval)

def get_profile(symbol: str) -> dict:
    if _is_indian_symbol(symbol):
        return _nse_get_profile(symbol)
    profile = _get_profile(symbol)
    if not profile.get("name") or profile["name"] == symbol:
        try: return _nse_get_profile(symbol)
        except: pass
    return profile

def get_metrics(symbol: str) -> dict:
    if _is_indian_symbol(symbol):
        return _nse_get_metrics(symbol)
    metrics = _get_metrics(symbol)
    if not metrics.get("eps"):
        try:
            nse_m = _nse_get_metrics(symbol)
            if nse_m.get("eps") or nse_m.get("pe_ratio"): return nse_m
        except: pass
    return metrics


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="TradeSense AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = """You are TradeSense AI, an elite financial trading assistant with deep expertise in global markets.

You ONLY respond to questions about:
- Stock markets, equity trading, shares, IPOs
- ETFs, mutual funds, index funds
- Forex (currency trading)
- Commodities (gold, oil, agricultural products)
- Bonds, fixed income, treasury securities
- Derivatives: options, futures, swaps
- Technical analysis: candlesticks, RSI, MACD, Bollinger Bands, support/resistance
- Fundamental analysis: P/E ratio, EPS, revenue, balance sheets
- Trading strategies: day trading, swing trading, value investing, momentum
- Portfolio management, diversification, risk management
- Cryptocurrency markets and trading
- Market terminology, order types, broker concepts
- Economic indicators affecting markets (CPI, GDP, Fed rates, etc.)

If a question is completely outside this domain, respond EXACTLY with:
"🚫 That question falls outside my trading domain. I'm TradeSense AI — your specialized trading assistant. Ask me about stocks, markets, technical analysis, trading strategies, or investment concepts!"

Format responses with clear markdown structure. Be concise but comprehensive."""

# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class CompareRequest(BaseModel):
    symbols: List[str]
    period: str = "3mo"

# ── Helper ─────────────────────────────────────────────────────────────────────
def fmt_cap(cap, symbol=""):
    if not cap: return "N/A"
    currency = "₹" if _is_indian_symbol(symbol) else "$"
    if cap >= 1e12: return f"{currency}{cap/1e12:.2f}T"
    if cap >= 1e9:  return f"{currency}{cap/1e9:.2f}B"
    return f"{currency}{cap/1e6:.2f}M"

# ── API: Chat streaming ────────────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured. Set the GROQ_API_KEY environment variable")

    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    async def generate():
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=1024, temperature=0.3, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = str(e)
            if "401" in err or "invalid_api_key" in err:
                yield f"data: {json.dumps({'error': '❌ Invalid Groq API Key. Get a valid key from https://console.groq.com and set the GROQ_API_KEY environment variable'})}\n\n"
            elif "429" in err or "rate" in err.lower():
                yield f"data: {json.dumps({'error': '⏳ Groq rate limit hit. Please wait a moment and try again.'})}\n\n"
            else:
                yield f"data: {json.dumps({'error': err})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ── API: Single stock ──────────────────────────────────────────────────────────
@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str, period: str = "3mo", interval: str = "1d"):
    try:
        symbol  = symbol.upper().strip()
        df      = get_candles(symbol, period, interval)
        profile = get_profile(symbol)
        metrics = get_metrics(symbol)

        ohlcv = [{"date": row["date"],
                  "open":   round(float(row["open"]),   4),
                  "high":   round(float(row["high"]),   4),
                  "low":    round(float(row["low"]),    4),
                  "close":  round(float(row["close"]),  4),
                  "volume": int(row["volume"])}
                 for _, row in df.iterrows()]

        closes  = [c["close"] for c in ohlcv]
        highs   = [c["high"]  for c in ohlcv]
        lows    = [c["low"]   for c in ohlcv]
        volumes = [c["volume"] for c in ohlcv]

        # ── SMA ──────────────────────────────────────────────────────
        sma20 = [round(float(np.mean(closes[max(0,i-19):i+1])), 4) for i in range(len(closes))]
        sma50 = [round(float(np.mean(closes[max(0,i-49):i+1])), 4) if i >= 19 else None for i in range(len(closes))]

        # ── EMA helper ───────────────────────────────────────────────
        def _ema(src, period):
            k = 2 / (period + 1)
            out = [None] * len(src)
            if len(src) >= period:
                out[period-1] = float(np.mean(src[:period]))
                for i in range(period, len(src)):
                    out[i] = round(src[i] * k + out[i-1] * (1 - k), 4)
            return out

        ema9  = _ema(closes, 9)
        ema21 = _ema(closes, 21)

        # ── Bollinger Bands (20, 2σ) ─────────────────────────────────
        bb_upper, bb_lower, bb_mid = [], [], []
        for i in range(len(closes)):
            window = closes[max(0, i-19):i+1]
            if len(window) >= 20:
                m = float(np.mean(window)); s = float(np.std(window))
                bb_mid.append(round(m, 4)); bb_upper.append(round(m+2*s, 4)); bb_lower.append(round(m-2*s, 4))
            else:
                bb_mid.append(None); bb_upper.append(None); bb_lower.append(None)

        # ── RSI (14) ──────────────────────────────────────────────────
        rsi = [None] * len(closes)
        if len(closes) > 14:
            deltas = np.diff(closes)
            gains  = np.where(deltas > 0, deltas, 0).astype(float)
            losses = np.where(deltas < 0, -deltas, 0).astype(float)
            avg_g  = float(np.mean(gains[:14]))
            avg_l  = float(np.mean(losses[:14]))
            for i in range(14, len(closes)):
                if i - 14 > 0:
                    avg_g = (avg_g * 13 + gains[i-1]) / 14
                    avg_l = (avg_l * 13 + losses[i-1]) / 14
                rs = avg_g / avg_l if avg_l != 0 else 0
                rsi[i] = round(100 - 100 / (1 + rs), 2)

        # ── MACD (12, 26, 9) ─────────────────────────────────────────
        ema12 = _ema(closes, 12); ema26 = _ema(closes, 26)
        macd_line  = [round(a-b, 4) if a is not None and b is not None else None for a,b in zip(ema12, ema26)]
        signal_line = [None] * len(macd_line)
        valid_idx   = [i for i, v in enumerate(macd_line) if v is not None]
        if len(valid_idx) >= 9:
            sig_vals = _ema([macd_line[i] for i in valid_idx], 9)
            for j, vi in enumerate(valid_idx): signal_line[vi] = sig_vals[j]
        macd_hist = [round(m-s, 4) if m is not None and s is not None else None for m,s in zip(macd_line, signal_line)]

        # ── VWAP ──────────────────────────────────────────────────────
        vwap = []
        cum_pv = cum_v = 0.0
        for c in ohlcv:
            typ_p = (c["high"] + c["low"] + c["close"]) / 3
            cum_pv += typ_p * c["volume"]; cum_v += c["volume"]
            vwap.append(round(cum_pv / cum_v, 4) if cum_v > 0 else None)

        cur     = closes[-1]
        prev    = closes[-2] if len(closes) > 1 else cur
        chg     = round(cur - prev, 4)
        chg_pct = round((chg / prev) * 100, 2) if prev else 0
        mktcap  = profile.get("market_cap")
        eps     = metrics.get("eps")
        pe      = metrics.get("pe_ratio") or (round(cur / eps, 2) if eps and eps > 0 else None)

        return {
            "symbol": symbol,
            "name":   profile.get("name", symbol),
            "current_price": cur,
            "change": chg, "change_pct": chg_pct,
            "ohlcv": ohlcv,
            "sma20": sma20, "sma50": sma50,
            "ema9": ema9, "ema21": ema21,
            "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
            "rsi": rsi,
            "macd_line": macd_line, "signal_line": signal_line, "macd_hist": macd_hist,
            "vwap": vwap,
            "info": {
                "market_cap":    mktcap,
                "market_cap_fmt": fmt_cap(mktcap, symbol),
                "pe_ratio":      pe,
                "eps":           eps,
                "week_52_high":  max(c["high"] for c in ohlcv) if ohlcv else None,
                "week_52_low":   min(c["low"]  for c in ohlcv) if ohlcv else None,
                "sector":        profile.get("sector", ""),
                "dividend_yield":None,
            }
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ── API: Comparison ────────────────────────────────────────────────────────────
@app.post("/api/compare")
async def compare_stocks(req: CompareRequest):
    try:
        symbols = [s.upper().strip() for s in req.symbols[:5]]
        results = []
        for symbol in symbols:
            try:
                df      = get_candles(symbol, req.period, "1d")
                profile = get_profile(symbol)
                metrics = get_metrics(symbol)

                closes  = df["close"].astype(float).tolist()
                volumes = df["volume"].astype(float).tolist()
                dates   = df["date"].tolist()
                sp, ep  = closes[0], closes[-1]
                ret     = round(((ep - sp) / sp) * 100, 2)

                daily  = pd.Series(closes).pct_change().dropna()
                vol    = round(float(daily.std() * np.sqrt(252) * 100), 2)
                rf     = 4.5 / 252
                excess = daily - rf / 100
                sharpe = round(float(excess.mean() / daily.std() * np.sqrt(252)), 3) if daily.std() > 0 else 0
                cum    = (1 + daily).cumprod()
                dd     = ((cum - cum.expanding().max()) / cum.expanding().max())
                max_dd = round(float(dd.min() * 100), 2)
                norm   = [round((p / sp) * 100, 4) for p in closes]
                mktcap = profile.get("market_cap")
                eps    = metrics.get("eps")
                pe     = metrics.get("pe_ratio") or (round(ep / eps, 2) if eps and eps > 0 else None)

                results.append({
                    "symbol": symbol,
                    "name":   profile.get("name", symbol),
                    "start_price": round(sp, 2), "end_price": round(ep, 2),
                    "total_return": ret, "volatility": vol,
                    "sharpe_ratio": sharpe, "max_drawdown": max_dd,
                    "avg_volume": round(float(np.mean(volumes))),
                    "market_cap": mktcap, "market_cap_fmt": fmt_cap(mktcap, symbol),
                    "pe_ratio": pe,
                    "dates": dates, "closes": [round(c,2) for c in closes],
                    "normalized": norm,
                })
            except Exception as e:
                print(f"  Skipping {symbol}: {e}")
                continue

        if not results:
            raise HTTPException(status_code=404, detail="No valid stock data found")

        if len(results) > 1:
            rets  = [r["total_return"] for r in results]
            vols  = [r["volatility"]   for r in results]
            sharps= [r["sharpe_ratio"] for r in results]
            max_r = max(rets)-min(rets) or 1
            max_v = max(vols)-min(vols) or 1
            max_s = max(sharps)-min(sharps) or 1
            def score(r):
                return ((r["total_return"]-min(rets))/max_r*0.45 +
                        (max(vols)-r["volatility"])/max_v*0.30 +
                        (r["sharpe_ratio"]-min(sharps))/max_s*0.25)
            results.sort(key=score, reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i+1; r["is_best"] = i==0

        return {"stocks": results, "period": req.period}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# ── API: Health ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok",
            "groq_configured":    bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"),
            "polygon_configured": bool(POLYGON_API_KEY and POLYGON_API_KEY != "your_polygon_api_key_here"),
            "timestamp": datetime.utcnow().isoformat()}

# ══════════════════════════════════════════════════════════════════════════════
#  FRONTEND HTML  (served directly from Python — no separate HTML file needed)
# ══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>TradeSense AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1/dist/chartjs-chart-financial.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root{
  --bg:#050a14;--surface:#0c1424;--surface2:#111d2e;--surface3:#162338;
  --border:rgba(99,179,237,0.15);--border2:rgba(99,179,237,0.28);
  --accent:#38bdf8;--accent2:#818cf8;--accent3:#34d399;
  --gold:#fbbf24;--red:#f87171;--green:#34d399;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --radius:14px;--r8:8px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:
  radial-gradient(ellipse 80% 40% at 20% 0%,rgba(56,189,248,.07) 0%,transparent 60%),
  radial-gradient(ellipse 60% 60% at 80% 100%,rgba(129,140,248,.05) 0%,transparent 60%);
  pointer-events:none;z-index:0}
body::after{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(56,189,248,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.02) 1px,transparent 1px);
  background-size:60px 60px;pointer-events:none;z-index:0}

/* ── LAYOUT ── */
.wrap{max-width:1440px;margin:0 auto;padding:0 20px;position:relative;z-index:1}

/* ── HEADER ── */
header{background:rgba(12,20,36,.92);backdrop-filter:blur(20px);
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:200}
.hdr-inner{display:flex;align-items:center;gap:12px;padding:12px 20px}
.logo-box{width:38px;height:38px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:14px;color:#fff;flex-shrink:0;
  box-shadow:0 0 20px rgba(56,189,248,.4)}
.logo-text{font-size:18px;font-weight:700;color:var(--text)}
.logo-sub{font-size:11px;color:var(--text3);margin-left:4px}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:16px}
.status-dot{font-size:12px;display:flex;align-items:center;gap:5px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
.dot.off{background:var(--red);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── TAB NAV ── */
.tab-nav{display:flex;background:var(--surface);border-bottom:1px solid var(--border);padding:0 20px;position:sticky;top:63px;z-index:100}
.tab-btn{padding:14px 22px;font-size:13px;font-weight:600;color:var(--text3);
  background:none;border:none;cursor:pointer;border-bottom:2px solid transparent;
  transition:all .2s;white-space:nowrap}
.tab-btn:hover{color:var(--text);background:rgba(56,189,248,.05)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

/* ── PANELS ── */
.panel{display:none;padding:20px 20px 40px;min-height:calc(100vh - 120px)}
.panel.active{display:block}

/* ── CARDS ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px}
.card-sm{background:var(--surface);border:1px solid var(--border);border-radius:var(--r8);padding:14px}

/* ── BUTTONS ── */
.btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border:none;
  border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .18s}
.btn-accent{background:var(--accent);color:#05131f}
.btn-accent:hover{background:#7dd3fc;transform:translateY(-1px)}
.btn-green{background:var(--accent3);color:#05131f}
.btn-green:hover{background:#6ee7b7}
.btn-ghost{background:var(--surface3);color:var(--text2)}
.btn-ghost:hover{background:var(--surface2);color:var(--text)}

/* ── INPUTS ── */
.inp{background:var(--surface3);border:1px solid var(--border);border-radius:8px;
  color:var(--text);padding:9px 13px;font-size:13px;outline:none;transition:.18s}
.inp:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(56,189,248,.12)}
.inp::placeholder{color:var(--text3)}
select.inp{cursor:pointer}

/* ── CHIPS ── */
.chip{display:inline-block;padding:5px 12px;background:var(--surface3);
  color:var(--accent);border-radius:20px;font-size:11px;cursor:pointer;
  border:1px solid var(--border);transition:.15s}
.chip:hover{background:var(--surface2);border-color:var(--accent)}

/* ════════ TAB 1 — CHAT ════════ */
.chat-layout{display:grid;grid-template-rows:auto 1fr auto;height:calc(100vh - 165px);gap:12px}
.chat-suggestions{display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.chat-window{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:75%;padding:12px 16px;border-radius:12px;font-size:13.5px;line-height:1.65;word-break:break-word}
.msg.user{align-self:flex-end;background:var(--accent2);color:#fff;border-bottom-right-radius:3px}
.msg.ai{align-self:flex-start;background:var(--surface3);color:var(--text);border-bottom-left-radius:3px}
.msg.error{align-self:flex-start;background:#2d0a0a;color:var(--red);border:1px solid rgba(248,113,113,.2)}
.msg-meta{font-size:10px;color:var(--text3);margin-bottom:4px}
.msg.user .msg-meta{text-align:right}
.msg h1,.msg h2,.msg h3{font-size:14px;margin:8px 0 4px;color:var(--accent)}
.msg ul,.msg ol{padding-left:18px;margin:6px 0}
.msg li{margin:3px 0}
.msg strong{color:var(--gold)}
.msg code{background:rgba(0,0,0,.3);padding:1px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:12px}
.typing-dots{display:flex;gap:5px;padding:14px 16px;background:var(--surface3);border-radius:12px;border-bottom-left-radius:3px;width:fit-content}
.typing-dots span{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:bounce .8s infinite}
.typing-dots span:nth-child(2){animation-delay:.15s}
.typing-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}}
.chat-input-row{display:flex;gap:10px}
.chat-input-row .inp{flex:1;padding:11px 14px}

/* ════════ TAB 2 — CHART ════════ */
.chart-controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:14px}
.chart-controls label{font-size:12px;color:var(--text2)}
.stats-strip{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:14px}
.stat-item{display:flex;flex-direction:column;gap:2px}
.stat-label{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.6px}
.stat-value{font-size:15px;font-weight:700;color:var(--text)}
.stat-value.up{color:var(--green)}
.stat-value.down{color:var(--red)}
.chart-area{position:relative;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px;min-height:420px}
.chart-placeholder{display:flex;align-items:center;justify-content:center;height:400px;
  color:var(--text3);font-size:15px;flex-direction:column;gap:10px}
.chart-placeholder span{font-size:36px}

/* ════════ TAB 3 — COMPARE ════════ */
.compare-controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:16px}
.sym-inputs{display:flex;flex-wrap:wrap;gap:8px}
.sym-inputs .inp{width:90px}
.rank-cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.rank-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px;min-width:175px;flex:1;transition:.2s;position:relative;overflow:hidden}
.rank-card.best{border-color:var(--gold);box-shadow:0 0 24px rgba(251,191,36,.12)}
.rank-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--border)}
.rank-card.best::before{background:linear-gradient(90deg,var(--gold),#f59e0b)}
.rank-badge{font-size:11px;font-weight:700;color:var(--text3);margin-bottom:6px}
.rank-badge.gold{color:var(--gold)}
.rank-sym{font-size:22px;font-weight:800;margin-bottom:2px}
.rank-name{font-size:11px;color:var(--text3);margin-bottom:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rank-ret{font-size:18px;font-weight:700;margin-bottom:12px}
.rank-ret.up{color:var(--green)}
.rank-ret.down{color:var(--red)}
.rank-metrics{display:flex;flex-direction:column;gap:5px}
.rank-metric{display:flex;justify-content:space-between;font-size:11.5px}
.rank-metric .lbl{color:var(--text3)}
.rank-metric .val{color:var(--text);font-weight:600}
.best-badge{margin-top:10px;text-align:center;font-size:11px;color:var(--gold);font-weight:700}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.chart-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.chart-box.full{grid-column:1/-1}
.chart-box h3{font-size:12px;color:var(--text2);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}
.metrics-table{width:100%;border-collapse:collapse;font-size:12px}
.metrics-table th{background:var(--surface3);color:var(--text3);padding:9px 12px;text-align:left;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.metrics-table td{padding:9px 12px;border-bottom:1px solid rgba(99,179,237,.06)}
.metrics-table tr:hover td{background:rgba(56,189,248,.04)}
.metrics-table .sym-cell{font-weight:700}
.metrics-table .up{color:var(--green)}
.metrics-table .down{color:var(--red)}

/* ── STATUS BAR ── */
.status-bar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text3);padding:8px 0}
.status-bar.ok{color:var(--green)}
.status-bar.err{color:var(--red)}
.spinner{width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;display:none}
.spinner.active{display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── FOOTER ── */
footer{background:var(--surface);border-top:1px solid var(--border);
  text-align:center;padding:12px;font-size:11px;color:var(--text3)}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}

/* ── OVERLAY LOADING ── */
#page-loader{position:fixed;inset:0;background:var(--bg);z-index:9999;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;transition:opacity .5s}
#page-loader.hidden{opacity:0;pointer-events:none}
.loader-logo{font-size:48px;font-weight:900;color:var(--accent);letter-spacing:-2px}
.loader-bar{width:220px;height:3px;background:var(--surface3);border-radius:2px;overflow:hidden}
.loader-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
  border-radius:2px;animation:fill 1.2s ease forwards}
@keyframes fill{from{width:0}to{width:100%}}
</style>
</head>
<body>

<!-- Page loader -->
<div id="page-loader">
  <div class="loader-logo">TS</div>
  <p style="color:var(--text2);font-size:14px">TradeSense AI is starting...</p>
  <div class="loader-bar"><div class="loader-fill"></div></div>
</div>

<!-- Header -->
<header>
  <div class="hdr-inner wrap">
    <div class="logo-box">TS</div>
    <span class="logo-text">TradeSense AI</span>
    <span class="logo-sub">Intelligent Trading Platform</span>
    <div class="hdr-right">
      <div class="status-dot" id="apiStatus">
        <span class="dot off" id="statusDot"></span>
        <span id="statusText">Checking...</span>
      </div>
    </div>
  </div>
</header>

<!-- Tab nav -->
<nav class="tab-nav">
  <div class="wrap" style="display:flex;gap:0;padding:0">
    <button class="tab-btn active" onclick="switchTab('chat')">🤖 AI Trading Assistant</button>
    <button class="tab-btn" onclick="switchTab('chart')">📊 Live Stock Chart</button>
    <button class="tab-btn" onclick="switchTab('compare')">⚖️ Stock Comparison</button>
  </div>
</nav>

<!-- ════ TAB 1: CHAT ════ -->
<div id="tab-chat" class="panel active">
  <div class="wrap">
    <div class="chat-layout">

      <div>
        <div class="chat-suggestions" style="margin-bottom:10px">
          <span style="font-size:12px;color:var(--text3);margin-right:4px">Try:</span>
          <span class="chip" onclick="sendChip('What is RSI and how do I use it?')">What is RSI?</span>
          <span class="chip" onclick="sendChip('Explain MACD indicator')">Explain MACD</span>
          <span class="chip" onclick="sendChip('What is a candlestick pattern?')">Candlestick patterns</span>
          <span class="chip" onclick="sendChip('Best day trading strategies for beginners')">Day trading tips</span>
          <span class="chip" onclick="sendChip('How to calculate P/E ratio?')">P/E ratio</span>
          <span class="chip" onclick="sendChip('What is Sharpe ratio?')">Sharpe ratio</span>
          <span class="chip" onclick="sendChip('Explain support and resistance levels')">Support & Resistance</span>
        </div>
      </div>

      <div class="chat-window" id="chatWindow">
        <div class="msg ai">
          <div class="msg-meta">TradeSense AI</div>
          <div>👋 <strong>Welcome to TradeSense AI!</strong><br><br>
          I'm your specialized trading assistant powered by <strong>Groq LLaMA 3</strong>. Ask me anything about:<br><br>
          • 📈 Stock markets &amp; equities<br>
          • 📉 Technical analysis (RSI, MACD, Bollinger Bands)<br>
          • 💡 Trading strategies &amp; risk management<br>
          • 🌐 Crypto, Forex &amp; Commodities<br>
          • 📊 Fundamental analysis &amp; valuations</div>
        </div>
      </div>

      <div>
        <div class="chat-input-row">
          <input class="inp" id="chatInput" placeholder="Ask anything about trading, stocks, or markets..." onkeydown="if(event.key==='Enter')sendMessage()"/>
          <button class="btn btn-accent" id="sendBtn" onclick="sendMessage()">Send ➤</button>
          <button class="btn btn-ghost" onclick="clearChat()">Clear</button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- ════ TAB 2: CHART ════ -->
<div id="tab-chart" class="panel">
  <div class="wrap">

    <div class="card" style="margin-bottom:14px">
      <div class="chart-controls">
        <div>
          <label>Symbol</label>
          <input class="inp" id="chartSym" value="AAPL" style="width:100px;margin-left:8px" onkeydown="if(event.key==='Enter')loadChart()" placeholder="e.g. AAPL"/>
        </div>
        <div>
          <label>Period</label>
          <select class="inp" id="chartPeriod" style="margin-left:8px">
            <option value="5d">5 Days</option>
            <option value="1mo">1 Month</option>
            <option value="3mo" selected>3 Months</option>
            <option value="6mo">6 Months</option>
            <option value="1y">1 Year</option>
            <option value="2y">2 Years</option>
            <option value="5y">5 Years</option>
          </select>
        </div>
        <div>
          <label>Interval</label>
          <select class="inp" id="chartInterval" style="margin-left:8px">
            <option value="1d" selected>Daily</option>
            <option value="1wk">Weekly</option>
            <option value="1mo">Monthly</option>
          </select>
        </div>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showSMA20" checked> SMA 20
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showSMA50" checked> SMA 50
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showEMA9"> EMA 9
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showEMA21"> EMA 21
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showBB" checked> Bollinger
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showVWAP"> VWAP
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showRSI" checked> RSI
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
          <input type="checkbox" id="showMACD" checked> MACD
        </label>
        <button class="btn btn-accent" onclick="loadChart()">📊 Load Chart</button>
        <div class="spinner" id="chartSpinner"></div>
        <span id="chartStatus" class="status-bar"></span>
      </div>
    </div>

    <div class="stats-strip" id="statsStrip"></div>

    <div class="chart-area">
      <div class="chart-placeholder" id="chartPlaceholder">
        <span>📊</span>
        Enter a symbol above and click Load Chart
      </div>
      <canvas id="candleChart" style="display:none"></canvas>
    </div>

    <div class="chart-box" style="margin-top:14px">
      <h3>Volume</h3>
      <canvas id="volumeChart" height="80"></canvas>
    </div>

    <div class="chart-box" id="rsiBox" style="margin-top:14px;display:none">
      <h3>RSI (14) <span id="rsiValue" style="color:var(--accent);font-weight:700;margin-left:8px"></span></h3>
      <canvas id="rsiChart" height="80"></canvas>
    </div>

    <div class="chart-box" id="macdBox" style="margin-top:14px;display:none">
      <h3>MACD (12, 26, 9)</h3>
      <canvas id="macdChart" height="80"></canvas>
    </div>

  </div>
</div>

<!-- ════ TAB 3: COMPARE ════ -->
<div id="tab-compare" class="panel">
  <div class="wrap">

    <div class="card" style="margin-bottom:16px">
      <div class="compare-controls">
        <div class="sym-inputs" id="symInputs">
          <input class="inp" value="AAPL" placeholder="Symbol"/>
          <input class="inp" value="MSFT" placeholder="Symbol"/>
          <input class="inp" value="GOOGL" placeholder="Symbol"/>
          <input class="inp" value="TSLA" placeholder="Symbol"/>
          <input class="inp" value="NVDA" placeholder="Symbol"/>
        </div>
        <select class="inp" id="comparePeriod">
          <option value="1mo">1 Month</option>
          <option value="3mo" selected>3 Months</option>
          <option value="6mo">6 Months</option>
          <option value="1y">1 Year</option>
          <option value="2y">2 Years</option>
        </select>
        <button class="btn btn-green" onclick="runComparison()">⚖️ Compare Stocks</button>
        <div class="spinner" id="compareSpinner"></div>
        <span id="compareStatus" class="status-bar"></span>
      </div>
    </div>

    <div id="compareResults" style="display:none">
      <div class="rank-cards" id="rankCards"></div>
      <div class="chart-grid">
        <div class="chart-box full">
          <h3>Normalised Performance (Base = 100)</h3>
          <canvas id="perfChart" height="120"></canvas>
        </div>
        <div class="chart-box">
          <h3>Total Return (%)</h3>
          <canvas id="retChart" height="200"></canvas>
        </div>
        <div class="chart-box">
          <h3>Risk vs Return</h3>
          <canvas id="riskChart" height="200"></canvas>
        </div>
      </div>
      <div class="card">
        <h3 style="font-size:13px;color:var(--text2);margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px">Detailed Metrics</h3>
        <div style="overflow-x:auto">
          <table class="metrics-table" id="metricsTable"></table>
        </div>
      </div>
    </div>

  </div>
</div>

<footer>
  TradeSense AI &nbsp;·&nbsp; Data via yFinance &nbsp;·&nbsp; AI via Groq LLaMA 3 &nbsp;·&nbsp;
  For educational purposes only. Not financial advice.
</footer>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
const state = { chatMessages: [], candleChart: null, volumeChart: null, rsiChart: null, macdChart: null, perfChart: null, retChart: null, riskChart: null };
const COLORS = ['#38bdf8','#34d399','#fbbf24','#f87171','#818cf8'];

// ── Page load ─────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  setTimeout(() => {
    document.getElementById('page-loader').classList.add('hidden');
  }, 1400);
  checkHealth();
});

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const dot  = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (d.groq_configured) {
      dot.classList.remove('off'); text.textContent = 'Groq Connected';
    } else {
      dot.classList.add('off'); text.textContent = 'No API Key';
    }
  } catch { document.getElementById('statusText').textContent = 'Offline'; }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.querySelectorAll('.tab-btn')[{chat:0,chart:1,compare:2}[tab]].classList.add('active');
}

// ════════════════════════════════════════════════════════════════════
//  TAB 1 — CHAT
// ════════════════════════════════════════════════════════════════════
function sendChip(text) {
  document.getElementById('chatInput').value = text;
  sendMessage();
}

function appendMsg(role, html, isHtml=false) {
  const win = document.getElementById('chatWindow');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = role === 'user' ? 'You' : role === 'ai' ? 'TradeSense AI' : 'Error';
  div.appendChild(meta);
  const body = document.createElement('div');
  body.innerHTML = isHtml ? html : html;
  div.appendChild(body);
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
  return body;
}

function showTyping() {
  const win = document.getElementById('chatWindow');
  const div = document.createElement('div');
  div.id = 'typingIndicator';
  div.className = 'typing-dots';
  div.innerHTML = '<span></span><span></span><span></span>';
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const msg   = input.value.trim();
  if (!msg) return;

  input.value = '';
  document.getElementById('sendBtn').disabled = true;
  state.chatMessages.push({ role: 'user', content: msg });

  appendMsg('user', escapeHtml(msg));
  showTyping();

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: state.chatMessages.slice(-20) }),
    });

    hideTyping();
    const msgBody = appendMsg('ai', '');
    let full = '';

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      for (const line of text.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') break;
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) { msgBody.innerHTML = '<span style="color:var(--red)">Error: ' + escapeHtml(parsed.error) + '</span>'; break; }
          if (parsed.content) {
            full += parsed.content;
            msgBody.innerHTML = marked.parse(full);
            document.getElementById('chatWindow').scrollTop = document.getElementById('chatWindow').scrollHeight;
          }
        } catch {}
      }
    }
    state.chatMessages.push({ role: 'assistant', content: full });
  } catch (e) {
    hideTyping();
    appendMsg('error', 'Connection error: ' + e.message);
  }
  document.getElementById('sendBtn').disabled = false;
}

function clearChat() {
  state.chatMessages = [];
  const win = document.getElementById('chatWindow');
  win.innerHTML = '';
  appendMsg('ai', 'Chat cleared. Ask me anything about trading! 📊');
}

// ════════════════════════════════════════════════════════════════════
//  TAB 2 — CHART
// ════════════════════════════════════════════════════════════════════
async function loadChart() {
  const sym      = document.getElementById('chartSym').value.toUpperCase().trim();
  const period   = document.getElementById('chartPeriod').value;
  const interval = document.getElementById('chartInterval').value;
  if (!sym) return;

  const spinner = document.getElementById('chartSpinner');
  const status  = document.getElementById('chartStatus');
  spinner.classList.add('active');
  status.textContent = 'Fetching ' + sym + '...';
  status.className = 'status-bar';

  try {
    const r = await fetch(`/api/stock/${sym}?period=${period}&interval=${interval}`);
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const d = await r.json();
    renderChart(d);
    status.textContent = '✓ ' + sym + ' loaded';
    status.className = 'status-bar ok';
  } catch(e) {
    status.textContent = '✗ ' + e.message;
    status.className = 'status-bar err';
  }
  spinner.classList.remove('active');
}

function renderChart(d) {
  const strip = document.getElementById('statsStrip');
  const chgUp = d.change >= 0;
  const lastRsi = d.rsi ? d.rsi.filter(v => v !== null).slice(-1)[0] : null;
  const lastMacd = d.macd_line ? d.macd_line.filter(v => v !== null).slice(-1)[0] : null;
  const rsiColor = lastRsi === null ? 'var(--text)' : lastRsi > 70 ? 'var(--red)' : lastRsi < 30 ? 'var(--green)' : 'var(--gold)';
  strip.innerHTML = `
    <div class="stat-item"><div class="stat-label">${d.name}</div><div class="stat-value">${d.symbol}</div></div>
    <div class="stat-item"><div class="stat-label">Price</div><div class="stat-value">$${d.current_price.toFixed(2)}</div></div>
    <div class="stat-item"><div class="stat-label">Change</div><div class="stat-value ${chgUp?'up':'down'}">${chgUp?'▲':'▼'} ${Math.abs(d.change).toFixed(2)} (${Math.abs(d.change_pct).toFixed(2)}%)</div></div>
    <div class="stat-item"><div class="stat-label">52W High</div><div class="stat-value">$${d.info.week_52_high||'N/A'}</div></div>
    <div class="stat-item"><div class="stat-label">52W Low</div><div class="stat-value">$${d.info.week_52_low||'N/A'}</div></div>
    <div class="stat-item"><div class="stat-label">P/E Ratio</div><div class="stat-value">${d.info.pe_ratio?d.info.pe_ratio.toFixed(2):'N/A'}</div></div>
    <div class="stat-item"><div class="stat-label">RSI (14)</div><div class="stat-value" style="color:${rsiColor}">${lastRsi !== null ? lastRsi.toFixed(1) : 'N/A'}</div></div>
    <div class="stat-item"><div class="stat-label">MACD</div><div class="stat-value" style="color:${lastMacd > 0 ? 'var(--green)' : 'var(--red)'}">${lastMacd !== null ? lastMacd.toFixed(3) : 'N/A'}</div></div>
    <div class="stat-item"><div class="stat-label">Sector</div><div class="stat-value" style="font-size:12px">${d.info.sector||'N/A'}</div></div>
  `;

  document.getElementById('chartPlaceholder').style.display = 'none';
  const canvas = document.getElementById('candleChart');
  canvas.style.display = 'block';

  ['candleChart','volumeChart','rsiChart','macdChart'].forEach(id => {
    const c = document.getElementById(id);
    if (c) { const ch = Chart.getChart(c); if (ch) ch.destroy(); }
    state[id] = null;
  });

  const labels = d.ohlcv.map(c => new Date(c.date).getTime());
  const candleData = d.ohlcv.map(c => ({ x: new Date(c.date).getTime(), o: c.open, h: c.high, l: c.low, c: c.close }));

  const showSMA20 = document.getElementById('showSMA20').checked;
  const showSMA50 = document.getElementById('showSMA50').checked;
  const showEMA9  = document.getElementById('showEMA9').checked;
  const showEMA21 = document.getElementById('showEMA21').checked;
  const showBB    = document.getElementById('showBB').checked;
  const showVWAP  = document.getElementById('showVWAP').checked;
  const showRSI   = document.getElementById('showRSI').checked;
  const showMACD  = document.getElementById('showMACD').checked;

  const lineDs = (label, data, color, dash=[]) => ({
    label, type:'line',
    data: data.map((v,i) => ({ x: labels[i], y: v })),
    borderColor: color, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.2, borderDash: dash,
  });

  const datasets = [{ label: d.symbol, data: candleData, color: { up:'#34d399', down:'#f87171', unchanged:'#94a3b8' } }];
  if (showSMA20) datasets.push(lineDs('SMA 20', d.sma20, '#38bdf8'));
  if (showSMA50) datasets.push(lineDs('SMA 50', d.sma50, '#fbbf24', [4,3]));
  if (showEMA9)  datasets.push(lineDs('EMA 9',  d.ema9,  '#a78bfa'));
  if (showEMA21) datasets.push(lineDs('EMA 21', d.ema21, '#fb923c', [3,2]));
  if (showVWAP)  datasets.push(lineDs('VWAP',   d.vwap,  '#f472b6', [6,3]));
  if (showBB && d.bb_upper) {
    datasets.push({ label:'BB Upper', type:'line', data: d.bb_upper.map((v,i)=>({x:labels[i],y:v})), borderColor:'rgba(167,139,250,0.6)', borderWidth:1, pointRadius:0, fill:false, borderDash:[2,2], tension:0.2 });
    datasets.push({ label:'BB Mid',   type:'line', data: d.bb_mid.map((v,i)=>({x:labels[i],y:v})),   borderColor:'rgba(167,139,250,0.3)', borderWidth:1, pointRadius:0, fill:'-1', backgroundColor:'rgba(167,139,250,0.03)', tension:0.2 });
    datasets.push({ label:'BB Lower', type:'line', data: d.bb_lower.map((v,i)=>({x:labels[i],y:v})), borderColor:'rgba(167,139,250,0.6)', borderWidth:1, pointRadius:0, fill:'-1', backgroundColor:'rgba(167,139,250,0.05)', tension:0.2 });
  }

  const xScale = { type:'time', time:{ unit:'day', tooltipFormat:'MMM d, yyyy', displayFormats:{day:'MMM d',week:'MMM d',month:'MMM yyyy'} }, ticks:{color:'#64748b',maxTicksLimit:12,maxRotation:30}, grid:{color:'rgba(99,179,237,.08)'} };

  state.candleChart = new Chart(canvas, {
    type:'candlestick', data:{ datasets },
    options:{ responsive:true, maintainAspectRatio:true,
      plugins:{ legend:{ labels:{color:'#94a3b8',font:{size:11}} },
        tooltip:{ callbacks:{ label: ctx => ctx.raw.o !== undefined ? [`O:${ctx.raw.o}`,`H:${ctx.raw.h}`,`L:${ctx.raw.l}`,`C:${ctx.raw.c}`] : `${ctx.dataset.label}: ${ctx.raw.y?.toFixed(3)}` } } },
      scales:{ x: xScale, y:{ ticks:{color:'#64748b'}, grid:{color:'rgba(99,179,237,.08)'} } } }
  });

  // Volume
  const volColors = d.ohlcv.map(c => c.close >= c.open ? 'rgba(52,211,153,.65)' : 'rgba(248,113,113,.65)');
  state.volumeChart = new Chart(document.getElementById('volumeChart'), {
    type:'bar', data:{ labels, datasets:[{ label:'Volume', data:d.ohlcv.map(c=>c.volume), backgroundColor:volColors, borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#64748b',maxTicksLimit:12,maxRotation:30},grid:{color:'rgba(99,179,237,.08)'}},
        y:{ticks:{color:'#64748b', callback:v=>v>=1e9?(v/1e9).toFixed(1)+'B':v>=1e6?(v/1e6).toFixed(1)+'M':v},grid:{color:'rgba(99,179,237,.08)'}} } }
  });

  // RSI
  const rsiBox = document.getElementById('rsiBox');
  rsiBox.style.display = showRSI ? 'block' : 'none';
  if (showRSI && d.rsi) {
    state.rsiChart = new Chart(document.getElementById('rsiChart'), {
      type:'line',
      data:{ labels, datasets:[
        { label:'RSI 14', data:d.rsi.map((v,i)=>({x:labels[i],y:v})), borderColor:'#38bdf8', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3,
          segment:{ borderColor: ctx => { const v=ctx.p1.parsed.y; return v>70?'#f87171':v<30?'#34d399':'#38bdf8'; } } },
        { label:'OB 70', data:labels.map(x=>({x,y:70})), borderColor:'rgba(248,113,113,.4)', borderWidth:1, borderDash:[4,4], pointRadius:0, fill:false },
        { label:'OS 30', data:labels.map(x=>({x,y:30})), borderColor:'rgba(52,211,153,.4)',  borderWidth:1, borderDash:[4,4], pointRadius:0, fill:false },
      ]},
      options:{ responsive:true, maintainAspectRatio:true,
        plugins:{ legend:{labels:{color:'#94a3b8',font:{size:10}}}, tooltip:{callbacks:{label:ctx=>`RSI: ${ctx.raw.y?.toFixed(1)}`}} },
        scales:{ x:{type:'time',time:{unit:'day',displayFormats:{day:'MMM d'}},ticks:{color:'#64748b',maxTicksLimit:10,maxRotation:25},grid:{color:'rgba(99,179,237,.08)'}},
          y:{min:0,max:100,ticks:{color:'#64748b',stepSize:20},grid:{color:'rgba(99,179,237,.08)'}} } }
    });
    const lastR = d.rsi.filter(v=>v!==null).slice(-1)[0];
    if (lastR !== null) {
      const el = document.getElementById('rsiValue');
      el.textContent = lastR.toFixed(1) + (lastR>70?' ⚠ Overbought':lastR<30?' ⚠ Oversold':'');
      el.style.color = lastR>70?'var(--red)':lastR<30?'var(--green)':'var(--gold)';
    }
  }

  // MACD
  const macdBox = document.getElementById('macdBox');
  macdBox.style.display = showMACD ? 'block' : 'none';
  if (showMACD && d.macd_line) {
    const histColors = d.macd_hist.map(v => v===null?'transparent':v>=0?'rgba(52,211,153,.7)':'rgba(248,113,113,.7)');
    state.macdChart = new Chart(document.getElementById('macdChart'), {
      type:'bar',
      data:{ labels, datasets:[
        { label:'Histogram', data:d.macd_hist.map((v,i)=>({x:labels[i],y:v})), backgroundColor:histColors, borderWidth:0, type:'bar' },
        { label:'MACD',      data:d.macd_line.map((v,i)=>({x:labels[i],y:v})), borderColor:'#38bdf8', borderWidth:1.5, pointRadius:0, fill:false, type:'line', tension:0.3 },
        { label:'Signal',    data:d.signal_line.map((v,i)=>({x:labels[i],y:v})), borderColor:'#fbbf24', borderWidth:1.5, pointRadius:0, fill:false, type:'line', tension:0.3, borderDash:[3,2] },
      ]},
      options:{ responsive:true, maintainAspectRatio:true, plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}},
        scales:{ x:{type:'time',time:{unit:'day',displayFormats:{day:'MMM d'}},ticks:{color:'#64748b',maxTicksLimit:10,maxRotation:25},grid:{color:'rgba(99,179,237,.08)'}},
          y:{ticks:{color:'#64748b'},grid:{color:'rgba(99,179,237,.08)'}} } }
    });
  }
}

// ════════════════════════════════════════════════════════════════════
//  TAB 3 — COMPARE
// ════════════════════════════════════════════════════════════════════
async function runComparison() {
  const inputs  = document.querySelectorAll('#symInputs .inp');
  const symbols = [...inputs].map(i => i.value.trim().toUpperCase()).filter(Boolean);
  if (symbols.length < 2) { alert('Enter at least 2 symbols.'); return; }

  const spinner = document.getElementById('compareSpinner');
  const status  = document.getElementById('compareStatus');
  spinner.classList.add('active');
  status.textContent = 'Fetching data for ' + symbols.join(', ') + '...';
  status.className = 'status-bar';
  document.getElementById('compareResults').style.display = 'none';

  try {
    const r = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbols, period: document.getElementById('comparePeriod').value }),
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const d = await r.json();
    renderComparison(d.stocks);
    status.textContent = `✓ Compared ${d.stocks.length} stocks`;
    status.className = 'status-bar ok';
  } catch(e) {
    status.textContent = '✗ ' + e.message;
    status.className = 'status-bar err';
  }
  spinner.classList.remove('active');
}

function renderComparison(stocks) {
  // ── Rank cards ──
  const cards = document.getElementById('rankCards');
  cards.innerHTML = '';
  stocks.forEach((s, i) => {
    const up   = s.total_return >= 0;
    const best = s.is_best;
    cards.innerHTML += `
      <div class="rank-card ${best?'best':''}">
        <div class="rank-badge ${best?'gold':''}">#${s.rank} ${best?'🏆 Best Pick':''}</div>
        <div class="rank-sym" style="color:${COLORS[i]}">${s.symbol}</div>
        <div class="rank-name">${s.name}</div>
        <div class="rank-ret ${up?'up':'down'}">${up?'▲':'▼'} ${Math.abs(s.total_return).toFixed(2)}%</div>
        <div class="rank-metrics">
          <div class="rank-metric"><span class="lbl">Start</span><span class="val">$${s.start_price}</span></div>
          <div class="rank-metric"><span class="lbl">End</span><span class="val">$${s.end_price}</span></div>
          <div class="rank-metric"><span class="lbl">Volatility</span><span class="val">${s.volatility}%</span></div>
          <div class="rank-metric"><span class="lbl">Sharpe</span><span class="val">${s.sharpe_ratio}</span></div>
          <div class="rank-metric"><span class="lbl">Max DD</span><span class="val" style="color:var(--red)">${s.max_drawdown}%</span></div>
          <div class="rank-metric"><span class="lbl">P/E</span><span class="val">${s.pe_ratio?s.pe_ratio.toFixed(1):'N/A'}</span></div>
          <div class="rank-metric"><span class="lbl">Mkt Cap</span><span class="val">${s.market_cap_fmt}</span></div>
        </div>
      </div>`;
  });

  // ── Performance chart ──
  const perfCanvas = document.getElementById('perfChart');
  if (state.perfChart) state.perfChart.destroy();
  state.perfChart = new Chart(perfCanvas, {
    type: 'line',
    data: {
      labels: stocks[0].dates,
      datasets: stocks.map((s,i) => ({
        label: s.symbol, data: s.normalized,
        borderColor: COLORS[i], borderWidth: 2,
        pointRadius: 0, fill: false, tension: 0.2,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 10, maxRotation: 25 }, grid: { color: 'rgba(99,179,237,.08)' } },
        y: { ticks: { color: '#64748b', callback: v => v.toFixed(0) }, grid: { color: 'rgba(99,179,237,.08)' } }
      }
    }
  });

  // ── Return bar ──
  const retCanvas = document.getElementById('retChart');
  if (state.retChart) state.retChart.destroy();
  state.retChart = new Chart(retCanvas, {
    type: 'bar',
    data: {
      labels: stocks.map(s => s.symbol),
      datasets: [{
        label: 'Total Return %',
        data: stocks.map(s => s.total_return),
        backgroundColor: stocks.map(s => s.total_return >= 0 ? 'rgba(52,211,153,.8)' : 'rgba(248,113,113,.8)'),
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true, indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: 'rgba(99,179,237,.08)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
      }
    }
  });

  // ── Risk vs Return scatter ──
  const riskCanvas = document.getElementById('riskChart');
  if (state.riskChart) state.riskChart.destroy();
  state.riskChart = new Chart(riskCanvas, {
    type: 'scatter',
    data: {
      datasets: stocks.map((s,i) => ({
        label: s.symbol,
        data: [{ x: s.volatility, y: s.total_return }],
        backgroundColor: COLORS[i], pointRadius: 10, pointHoverRadius: 13,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { title: { display: true, text: 'Volatility (%)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(99,179,237,.08)' } },
        y: { title: { display: true, text: 'Return (%)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(99,179,237,.08)' } }
      }
    }
  });

  // ── Metrics table ──
  const tbl = document.getElementById('metricsTable');
  tbl.innerHTML = `
    <thead><tr>
      <th>Symbol</th><th>Start $</th><th>End $</th><th>Return</th>
      <th>Volatility</th><th>Sharpe</th><th>Max Drawdown</th>
      <th>Avg Volume</th><th>Market Cap</th><th>P/E</th>
    </tr></thead>
    <tbody>${stocks.map((s,i) => `<tr>
      <td class="sym-cell" style="color:${COLORS[i]}">${s.symbol}</td>
      <td>$${s.start_price}</td>
      <td>$${s.end_price}</td>
      <td class="${s.total_return>=0?'up':'down'}">${s.total_return>0?'+':''}${s.total_return}%</td>
      <td>${s.volatility}%</td>
      <td>${s.sharpe_ratio}</td>
      <td class="down">${s.max_drawdown}%</td>
      <td>${fmtNum(s.avg_volume)}</td>
      <td>${s.market_cap_fmt}</td>
      <td>${s.pe_ratio?s.pe_ratio.toFixed(1):'N/A'}</td>
    </tr>`).join('')}</tbody>`;

  document.getElementById('compareResults').style.display = 'block';
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtNum(n) {
  if (!n) return 'N/A';
  if (n>=1e9) return (n/1e9).toFixed(2)+'B';
  if (n>=1e6) return (n/1e6).toFixed(2)+'M';
  if (n>=1e3) return (n/1e3).toFixed(1)+'K';
  return n;
}
</script>
</body>
</html>"""

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_frontend(path: str = ""):
    return HTMLResponse(content=HTML)

# ── Launch ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, webbrowser, threading

    def open_browser():
        import time; time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "="*55)
    print("  🚀  TradeSense AI")
    print("="*55)
    print(f"  ✅  Open in browser: http://localhost:8000")
    groq_ok = GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here"
    poly_ok = POLYGON_API_KEY and POLYGON_API_KEY != "your_polygon_api_key_here"
    print(f"  {'✅' if groq_ok else '⚠️ '}  Groq API:    {'Configured' if groq_ok else 'NOT SET — chatbot disabled'}")
    print(f"  {'✅' if poly_ok else '⚠️ '}  Polygon API: {'Configured' if poly_ok else 'NOT SET — stock charts disabled'}")
    print(f"  ✅  NSEPython:   Active (use SBIN.NS, TCS.NS, RELIANCE.BO etc.)")
    print("  ℹ️   Press Ctrl+C to stop")
    print("="*55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
