# Luxverum Capital â€” Complete Command Reference

## Important: Always Run Through the Orchestrator

Use `run_trading_system.py` for daily operations â€” it runs the regime detector first,
which sets the correct trailing stop percentages, position sizing, and risk parameters.

Running modules directly (like `03_execution_engine.py --protect-only`) skips the regime
detector and uses default parameters instead of bull/bear-adjusted ones.

**Example of the difference:**
- Through orchestrator (bull market): 5.0% trail, 5.0% trigger
- Direct standalone (no regime): 4.0% trail, 5.0% trigger â† DLR got this today

---

## Daily Workflow (Recommended Order)

### Morning â€” Full Rebalance (Primary Daily Command)

```
python run_trading_system.py --rebalance --dry-run
```
Preview the full plan: regime detection â†’ AI analysis â†’ portfolio status â†’ rebalance targets â†’ trailing stop graduations. Review the output, then execute live:

```
python run_trading_system.py --rebalance
```
This does EVERYTHING: detects regime, runs both AI engines, scores positions, executes trades with brackets, audits protection, graduates winners to trailing stops.

### Midday â€” Quick Protection Check

```
python run_trading_system.py --protect
```
Fixes any unprotected positions + graduates new winners to trailing stops. Runs regime detector first so trail percentages are correct.

### Anytime â€” Read-Only Audit

```
python run_trading_system.py --audit
```
Just checks protection coverage. No changes. No trades.

---

## All Orchestrator Commands (run_trading_system.py)

| Command | What It Does |
|---------|-------------|
| `--rebalance` | Full pipeline: Regime â†’ AI Analysis â†’ Status â†’ Rebalance + Protect + Graduate |
| `--rebalance --dry-run` | Same but preview only â€” no live trades |
| `--full` | Regime â†’ AI Analysis â†’ Status â†’ New trades only (no rebalance) + Protect |
| `--full --dry-run` | Preview new trades only |
| `--protect` | Status â†’ Fix unprotected positions + Graduate winners |
| `--audit` | Read-only protection check |
| `--analyze` | Only run AI market analysis (Claude + GPT-4o) |
| `--status` | Only show portfolio status |
| `--snapshot` | Capture daily portfolio snapshot |
| `--options` | Run options strategies engine |
| `--options --strategy spreads protect` | Specific options strategies |
| `--backtest` | Run historical backtesting |

---

## Standalone Module Commands

These skip the orchestrator. Use when you need to run a specific module independently.

### Execution Engine (03_execution_engine.py)

| Command | What It Does |
|---------|-------------|
| `--rebalance --dry-run` | Preview full rebalance plan |
| `--rebalance` | Execute live rebalance |
| `--new-trades --dry-run` | Preview new position entries only |
| `--new-trades` | Execute new entries only |
| `--protect-only --dry-run` | Preview protection fixes + graduations |
| `--protect-only` | Fix protection + graduate winners (live) |
| `--graduate --dry-run` | Preview trailing stop graduations only |
| `--graduate` | Execute trailing stop graduations only |
| `--audit` | Read-only protection check |

**Warning:** Running these directly shows "No fresh regime context â€” using defaults" and uses default parameters instead of regime-adjusted ones. Always prefer `run_trading_system.py`.

### Verification Tool (verify_trailing_stops.py)

| Command | What It Does |
|---------|-------------|
| (no flags) | Full verification: position map + protection status + trailing stop details + history |
| `--orders` | Detailed breakdown of every open order |
| `--history` | Today's filled and canceled orders + last execution log |

### Daily Snapshot (08_daily_snapshot.py)

| Command | What It Does |
|---------|-------------|
| (no flags) | Full snapshot + reconciliation |
| `--pre-trade` | Capture pre-trade state |
| `--post-trade` | Capture post-trade state + reconcile |
| `--reconcile` | Compare planned vs actual trades |
| `--cancel-stale` | Cancel orphaned limit sells |
| `--cancel-stale --dry-run` | Preview stale order cleanup |
| `--history` | List available snapshot dates |

### Other Modules

| Module | Command | What It Does |
|--------|---------|-------------|
| `00_regime_detector.py` | (no flags) | Detect market regime (bull/bear/sideways) |
| `01_market_analysis.py` | (no flags) | Run AI analysis (Claude + GPT-4o) |
| `02_portfolio_status.py` | (no flags) | Show current portfolio |
| `06_options_engine.py` | `--strategy spreads` | Run options strategies |
| `07_backtest_engine.py` | (no flags) | Run backtesting |

---

## Quick Reference Card

### "I want to..." â†’ Run this:

| I want to... | Command |
|--------------|---------|
| Start my trading day | `python run_trading_system.py --rebalance --dry-run` then `--rebalance` |
| Check if all positions are protected | `python verify_trailing_stops.py` |
| See which stocks have trailing stops | `python verify_trailing_stops.py` |
| See today's filled orders | `python verify_trailing_stops.py --history` |
| Fix any unprotected positions | `python run_trading_system.py --protect` |
| Manually graduate winners to trailing stops | `python run_trading_system.py --protect` |
| Just see the AI recommendations | `python run_trading_system.py --analyze` |
| Take a portfolio snapshot for records | `python run_trading_system.py --snapshot` |
| Check order details on Alpaca | `python verify_trailing_stops.py --orders` |
| Run options strategies | `python run_trading_system.py --options --strategy spreads protect` |
| Backtest the system | `python run_trading_system.py --backtest` |
| Preview without executing anything | Add `--dry-run` to any command above |

---

## How the Trailing Stop System Works

### Position Lifecycle

```
BUY ENTRY
  â””â”€ Bracket: Market buy + SL (-7%) + TP (+20%)
       â”‚
       â”œâ”€ Price drops to SL â†’ SOLD (loss capped)
       â”‚
       â”œâ”€ Price hits TP (+20%) â†’ SOLD (static exit)
       â”‚
       â””â”€ Price gains +5% (trigger threshold)
            â”‚
            â””â”€ GRADUATION: Cancel bracket â†’ Trailing stop
                 â”‚
                 â”‚  Bull market: 5.0% trail
                 â”‚  Sideways:    4.0% trail
                 â”‚  Bear market: 3.0% trail
                 â”‚
                 â”œâ”€ Price keeps rising â†’ Trail follows (Alpaca auto)
                 â”‚    â”‚
                 â”‚    â””â”€ At +15%: TIGHTEN trail (bull: 3.5%, bear: 2.0%)
                 â”‚
                 â””â”€ Price reverses â†’ Trail stop triggers â†’ SOLD with locked gain
```

### When Does Graduation Run?

- Automatically during every `--rebalance` (after the protection audit step)
- Automatically during every `--protect` / `--protect-only`
- On demand with `--graduate`

### What Alpaca Handles vs What We Handle

| Alpaca (automatic, real-time) | Our System (runs when you execute) |
|------------------------------|-----------------------------------|
| Tracks high-water mark (HWM) for stocks | Decides WHEN to graduate (at +5% gain) |
| Ratchets stop price up for stocks | Decides trail WIDTH (regime-adaptive) |
| Triggers sell on pullback | Tightens trail at +15% gain |
| Crypto: executes stop_limit/limit orders | Crypto: simulates trailing by replacing stop_limit |
| All 24/7 for crypto simple orders | Logs everything for verification |

---

## Crypto Protection (Alpaca Limitations)

Alpaca crypto orders support **only**: `market`, `limit`, `stop_limit`

**NOT supported for crypto:** bracket, OCO, trailing_stop order classes.

### How We Handle It

| Order Type | Stocks | Crypto |
|-----------|--------|--------|
| Entry | Bracket (market + SL + TP) | Simple market + immediate stop_limit |
| Stop-Loss | OCO leg (auto-paired) | Separate stop_limit order |
| Take-Profit | OCO leg (auto-paired) | Separate limit sell (best-effort) |
| Trailing Stop | TrailingStopOrderRequest | Simulated: replace stop_limit at higher price |
| Protection Repair | Cancel + fresh OCO | Cancel + fresh stop_limit + limit |

### Crypto Position Lifecycle

```
BUY ENTRY (simple market order)
  └─ Immediate: stop_limit sell (downside protection)
  └─ Best-effort: limit sell (take-profit, may fail if qty locked)
       │
       ├─ Price drops to stop_limit trigger → SOLD (loss capped)
       │
       ├─ Price hits limit sell → SOLD (profit taken)
       │
       └─ Price gains +5% (trigger threshold)
            │
            └─ GRADUATION: Cancel stop_limit → New stop_limit at trail level
                 (simulated trailing: stop moves up as price rises)
                 Re-evaluated on each --protect / --graduate run
```

### Important Notes

- Crypto stop_limit orders include a 0.5% slippage buffer (limit = stop × 0.995)
- Since crypto has no true OCO, the stop and TP are independent orders
- Daily audit (`--protect`) cleans up orphaned orders if one side fills
- Crypto trailing is **simulated** — it only updates when you run `--protect` or `--graduate`
  (unlike stocks where Alpaca tracks the high-water mark automatically 24/7)

---

## Files in Your Trading System

```
Trader Version 1/
â”œâ”€â”€ config.py                    â† Shared config (API keys, regime loader)
â”œâ”€â”€ config.json                  â† Your API keys and settings
â”œâ”€â”€ run_trading_system.py        â† MAIN ORCHESTRATOR (use this)
â”œâ”€â”€ 00_regime_detector.py        â† Market regime detection
â”œâ”€â”€ 01_market_analysis.py        â† AI analysis (Claude + GPT-4o)
â”œâ”€â”€ 02_portfolio_status.py       â† Portfolio dashboard
â”œâ”€â”€ 03_execution_engine.py       â† Trade execution + protection + graduation
â”œâ”€â”€ 06_options_engine.py         â† Options strategies
â”œâ”€â”€ 07_backtest_engine.py        â† Historical backtesting
â”œâ”€â”€ 08_daily_snapshot.py         â† Snapshots + reconciliation
â”œâ”€â”€ verify_trailing_stops.py     â† NEW: Protection verification tool
â”œâ”€â”€ data/                        â† AI recommendations, regime context, logs
â”œâ”€â”€ logs/                        â† Execution logs
â””â”€â”€ snapshots/                   â† Daily portfolio snapshots
    â””â”€â”€ 2026-02-12/
        â”œâ”€â”€ pre_rebalance_*.json
        â””â”€â”€ post_rebalance_*.json
```

---

## Current Protection Parameters by Regime

| Parameter | Bull Strong | Bull | Sideways | Bear | Bear Strong |
|-----------|-------------|------|----------|------|-------------|
| Stop Loss | 8.0% | 7.0% | 5.0% | 4.0% | 3.0% |
| Take Profit | 25.0% | 20.0% | 12.0% | 8.0% | 5.0% |
| Trail Width | 5.0% | 5.0% | 4.0% | 3.0% | 3.0% |
| Graduate Trigger | +5.0% | +5.0% | +5.0% | +3.0% | +3.0% |
| Tighten Trigger | +15.0% | +15.0% | +15.0% | +15.0% | +15.0% |
| Tighten Trail To | 3.5% | 3.5% | 3.0% | 2.0% | 2.0% |
| Position Size | 7.0% | 5.0% | 3.5% | 2.5% | 1.5% |
| Max Positions | 25 | 20 | 15 | 10 | 5 |
| Cash Reserve | 2.0% | 3.0% | 10.0% | 20.0% | 40.0% |
| Confidence Min | 40% | 45% | 55% | 65% | 75% |