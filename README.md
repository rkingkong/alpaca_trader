# Luxverum Capital — AI-Driven Trading System

An automated, AI-driven investment system that uses dual-AI consensus (Claude + GPT-4o) for market analysis, systematic risk management, and automated execution through Alpaca's API.

## Architecture

```
run_trading_system.py          ← Orchestrator (single entry point)
  ├── 00_regime_detector.py    ← Market regime classification (bull/bear/sideways)
  ├── 01_market_analysis.py    ← Dual-AI opportunity discovery
  ├── 02_portfolio_status.py   ← Portfolio state from Alpaca
  ├── 03_execution_engine.py   ← Rebalance, execute, protect, graduate
  ├── 04_trade_sheet.py        ← Human-readable trade execution sheets
  ├── 06_options_engine.py     ← Bull spreads, PMCC, CSP, protective puts
  ├── 07_backtest_engine.py    ← Historical performance validation
  └── 08_daily_snapshot.py     ← Portfolio snapshots & reconciliation

config.py                      ← Shared config & API client factory
verify_trailing_stops.py       ← Protection verification utility
ARCHITECTURE.md                ← Complete command reference
```

## Key Features

- **Dual-AI Consensus**: Claude and GPT-4o independently analyze markets; consensus signals get conviction bonuses
- **Regime-Adaptive**: Parameters automatically adjust for bull, sideways, and bear markets
- **6-Layer Protection**: Bracket orders → OCO → trailing stops → graduation → staleness detection → circuit breakers
- **Conviction-Based Sizing**: Higher-confidence positions get larger allocations and wider risk parameters
- **Comprehensive Audit**: Validates stop qty coverage, stop price staleness, and protection quality

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rkingkong/alpaca_trader.git
cd alpaca_trader

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp config.json.example config.json
# Edit config.json with your real API keys

# 4. Preview a rebalance (dry run)
python run_trading_system.py --rebalance --dry-run

# 5. Execute live
python run_trading_system.py --rebalance
```

## Daily Workflow

| Command | Purpose |
|---------|---------|
| `--rebalance --dry-run` | Preview full pipeline |
| `--rebalance` | Execute: regime → AI analysis → rebalance → protect → graduate |
| `--protect` | Fix unprotected positions + graduate winners |
| `--audit` | Read-only protection check |
| `--status` | Current portfolio snapshot |
| `--options` | Run options strategies |

## API Keys Required

| Service | Purpose |
|---------|---------|
| Alpaca | Trading execution & market data |
| Anthropic | Claude AI analysis |
| OpenAI | GPT-4o analysis |
| Finnhub | Financial news & earnings |
| FRED | Macroeconomic indicators |
| Alpha Vantage | Sector performance data |

## Data Sources

The system aggregates data from 6+ sources: Alpaca (prices), Finnhub (news, earnings, analyst data), FRED (macro indicators), Alpha Vantage (sector rotation), Yahoo Finance (fundamentals, sector ETFs), and dual AI models for synthesis.

## Risk Management

- **Position-level**: Stop-loss + take-profit OCO brackets on every position
- **Trailing stops**: Profitable positions graduate from static brackets to dynamic trails
- **Portfolio-level**: Max positions limit, cash reserve target, daily loss circuit breaker
- **Protection audit**: Validates qty coverage, stop price freshness, and auto-repairs degraded stops

## License

Private — All rights reserved.

## Mission

Luxverum Capital is an AI-driven investment fund with a charitable mission to maximize trading profits to help children in Venezuela.