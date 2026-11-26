# TradeSense AI

A single-file financial analytics web application combining a domain-focused AI assistant, interactive stock charts and comparative risk analytics.

## Features
- Groq LLaMA 3.3 chatbot with server-sent event streaming
- Historical OHLCV charts, SMA, EMA, RSI, MACD, Bollinger Bands and VWAP
- Comparison of up to five stocks using returns, annualized volatility, Sharpe ratio and maximum drawdown
- Polygon, NSEPython and Yahoo Finance data integrations; five-minute Polygon cache

## Setup
Requires Python 3.10+ and internet access.

```bash
python -m pip install -r requirements.txt
```
Copy `.env.example` to `.env` and fill in your own API keys. Never commit `.env`.

```bash
python trade_bot.py
```
Open http://localhost:8000. API documentation: http://localhost:8000/docs.

## Architecture
FastAPI serves the embedded HTML/CSS/JavaScript interface and JSON/SSE endpoints. Pandas and NumPy calculate indicators and comparison metrics; Chart.js renders charts.

## Limitations
Data availability depends on provider access and rate limits; quotes may be delayed. The comparison ranking is a heuristic, not a prediction. Indicator calculations and frontend security require further validation before public deployment. This is an educational portfolio project, not financial advice. GitHub Pages cannot run its Python backend.
