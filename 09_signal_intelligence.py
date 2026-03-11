#!/usr/bin/env python3
"""
Module 9: Signal Intelligence Engine
======================================
Complete feedback loop that transforms the trading system from stateless
AI calls into a self-calibrating, learning system.

Capabilities:
  1. SIGNAL TRACKER     - Logs every AI recommendation + tracks price outcomes
  2. WEIGHTED CONSENSUS - Dynamic AI weighting based on historical hit rates
  3. REGIME CONSTRAINTS  - Generates regime-specific trading rules from data
  4. POST-MORTEM AGENT  - AI analysis of closed trades (weekly review)
  5. PATTERN LIBRARY    - Extracts winning technical signatures from history

Data Flow:
  Module 01 (recommendations.json) ──┐
  Module 02 (portfolio_status.json) ──┤
  Module 00 (regime_context.json)  ───┼──► 09_signal_intelligence.py
  Module 08 (snapshots/)           ───┤        │
  Alpaca API (price lookups)       ───┘        ▼
                                         data/signal_log.json
                                         data/signal_performance.json
                                         data/ai_feedback_context.json
                                         data/pattern_library.json
                                         data/regime_constraints.json
                                               │
                                               ▼
                                    Module 01 reads ai_feedback_context.json
                                    and injects into Claude/GPT prompts

Usage:
  python 09_signal_intelligence.py --log          # Log today's signals + update outcomes
  python 09_signal_intelligence.py --stats         # Compute performance stats
  python 09_signal_intelligence.py --feedback      # Generate AI feedback context
  python 09_signal_intelligence.py --postmortem    # Run AI post-mortem on closed trades
  python 09_signal_intelligence.py --patterns      # Build/update pattern library
  python 09_signal_intelligence.py --full          # All of the above in sequence
  python 09_signal_intelligence.py --report        # Print human-readable report
"""

import os
import sys
import json
import re
import argparse
import time
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import traceback

import numpy as np
import pandas as pd

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Shared config
try:
    from config import (
        Config, Clients, load_regime_context, RegimeContext,
        DATA_DIR, LOGS_DIR, SNAPSHOTS_DIR,
        load_json, save_json, print_header, print_step,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        Config, Clients, load_regime_context, RegimeContext,
        DATA_DIR, LOGS_DIR, SNAPSHOTS_DIR,
        load_json, save_json, print_header, print_step,
    )

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# CONSTANTS
# =============================================================================

SIGNAL_LOG_FILE = "signal_log.json"
SIGNAL_PERF_FILE = "signal_performance.json"
FEEDBACK_CONTEXT_FILE = "ai_feedback_context.json"
PATTERN_LIBRARY_FILE = "pattern_library.json"
REGIME_CONSTRAINTS_FILE = "regime_constraints.json"

# Outcome check windows (trading days)
OUTCOME_WINDOWS = [1, 5, 20]

# Minimum signals needed before stats are meaningful
MIN_SIGNALS_FOR_STATS = 10
MIN_SIGNALS_PER_BUCKET = 5


# =============================================================================
# SIGNAL TRACKER (Feature #1)
# =============================================================================

class SignalTracker:
    """
    Logs every AI recommendation and tracks price outcomes over time.
    
    Each signal entry:
    {
        "signal_id": "hash",
        "date": "2026-03-10",
        "timestamp": "2026-03-10T12:18:24",
        "symbol": "OXY",
        "action": "BUY",
        "confidence": 0.85,
        "ai_agreement": "full",
        "source_claude_conf": 0.90,
        "source_gpt_conf": 0.80,
        "conviction_tier": "high",
        "sector": "Energy",
        "technical_bias": "bullish",
        "regime": "sideways",
        "regime_score": 0.17,
        "entry_price": 52.91,
        "suggested_sl_pct": 6.0,
        "suggested_tp_pct": 18.0,
        "indicators": {...},
        "outcomes": {
            "1d": {"price": 53.20, "return_pct": 0.55, "checked": "2026-03-11"},
            "5d": {"price": 54.10, "return_pct": 2.25, "checked": "2026-03-17"},
            "20d": {"price": null, "return_pct": null, "checked": null}
        },
        "hit_sl": false,
        "hit_tp": false,
        "max_favorable_excursion_pct": 4.2,
        "max_adverse_excursion_pct": -1.1,
        "was_executed": true,
        "execution_pnl_pct": 2.5  // actual portfolio P&L if executed
    }
    """

    def __init__(self):
        self.log_path = os.path.join(DATA_DIR, SIGNAL_LOG_FILE)
        self.signals = self._load_log()
        self.cfg = Config()

        # Lazy clients
        self._stock_client = None
        self._crypto_client = None
        self._trading_client = None

    def _load_log(self) -> List[Dict]:
        """Load existing signal log."""
        try:
            with open(self.log_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_log(self):
        """Persist signal log to disk."""
        with open(self.log_path, 'w') as f:
            json.dump(self.signals, f, indent=2)

    @property
    def stock_client(self):
        if self._stock_client is None:
            self._stock_client = StockHistoricalDataClient(
                self.cfg.ALPACA_API_KEY, self.cfg.ALPACA_SECRET_KEY
            )
        return self._stock_client

    @property
    def crypto_client(self):
        if self._crypto_client is None:
            self._crypto_client = CryptoHistoricalDataClient(
                self.cfg.ALPACA_API_KEY, self.cfg.ALPACA_SECRET_KEY
            )
        return self._crypto_client

    @property
    def trading_client(self):
        if self._trading_client is None:
            self._trading_client = TradingClient(
                self.cfg.ALPACA_API_KEY, self.cfg.ALPACA_SECRET_KEY,
                paper=self.cfg.PAPER_TRADING
            )
        return self._trading_client

    def _make_signal_id(self, date_str: str, symbol: str, action: str) -> str:
        """Deterministic signal ID to prevent duplicates."""
        raw = f"{date_str}|{symbol}|{action}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def log_todays_signals(self) -> int:
        """
        Read today's recommendations.json and log all signals.
        Skips signals already logged (idempotent).
        """
        recs = load_json("recommendations.json")
        if not recs:
            print("  No recommendations.json found")
            return 0

        # Load regime context
        regime_ctx = load_json("regime_context.json") or {}
        regime_name = regime_ctx.get("regime", "unknown")
        regime_score = regime_ctx.get("regime_score", 0)

        # Load portfolio to check what was actually executed
        portfolio = load_json("portfolio_status.json") or {}
        held_symbols = set()
        positions = portfolio.get("positions", {})
        if isinstance(positions, dict):
            held_symbols = set(positions.keys())
        elif isinstance(positions, list):
            held_symbols = {p.get("symbol", "") for p in positions}

        today = datetime.now().strftime("%Y-%m-%d")
        existing_ids = {s["signal_id"] for s in self.signals}
        logged = 0

        # Process both buy and sell signals
        all_signals = []
        for sig in recs.get("buy_signals", []):
            sig["_action_type"] = "BUY"
            all_signals.append(sig)
        for sig in recs.get("sell_signals", []):
            sig["_action_type"] = "SELL"
            all_signals.append(sig)
        for sig in recs.get("watch", []):
            sig["_action_type"] = "WATCH"
            all_signals.append(sig)

        for sig in all_signals:
            symbol = sig.get("symbol", "")
            action = sig.get("_action_type", sig.get("action", "UNKNOWN"))
            if not symbol:
                continue

            signal_id = self._make_signal_id(today, symbol, action)
            if signal_id in existing_ids:
                continue

            # Extract per-AI confidence if available
            claude_conf = sig.get("claude_confidence", sig.get("confidence", 0))
            gpt_conf = sig.get("chatgpt_confidence", sig.get("confidence", 0))

            entry = {
                "signal_id": signal_id,
                "date": today,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "confidence": sig.get("confidence", 0),
                "ai_agreement": sig.get("ai_agreement", "single"),
                "source_claude_conf": claude_conf,
                "source_gpt_conf": gpt_conf,
                "conviction_tier": sig.get("conviction_tier", "base"),
                "sector": sig.get("sector", "Unknown"),
                "technical_bias": sig.get("technical_bias", "neutral"),
                "regime": regime_name,
                "regime_score": regime_score,
                "entry_price": sig.get("current_price", sig.get("price", None)),
                "suggested_sl_pct": sig.get("suggested_stop_loss_pct",
                                            sig.get("stop_loss_pct", None)),
                "suggested_tp_pct": sig.get("suggested_take_profit_pct",
                                            sig.get("take_profit_pct", None)),
                "rs_vs_spy": sig.get("rs_vs_spy",
                                     sig.get("indicators", {}).get("rs_vs_spy", None)),
                "momentum_1m": sig.get("indicators", {}).get("momentum_1m_pct", None),
                "rsi": sig.get("indicators", {}).get("rsi", None),
                "above_sma50": sig.get("indicators", {}).get("above_sma50", None),
                "above_sma200": sig.get("indicators", {}).get("above_sma200", None),
                "volume_ratio": sig.get("indicators", {}).get("volume_ratio", None),
                "outcomes": {
                    str(w): {"price": None, "return_pct": None, "checked": None}
                    for w in OUTCOME_WINDOWS
                },
                "hit_sl": False,
                "hit_tp": False,
                "max_favorable_excursion_pct": None,
                "max_adverse_excursion_pct": None,
                "was_executed": self._check_if_executed(symbol, action, held_symbols),
                "execution_pnl_pct": None,
            }

            self.signals.append(entry)
            existing_ids.add(signal_id)
            logged += 1

        self._save_log()
        return logged

    def _check_if_executed(self, symbol: str, action: str, held_symbols: set) -> bool:
        """Check if signal was actually traded (heuristic)."""
        # Normalize crypto symbols
        norm = symbol.replace("/", "")
        if action == "BUY":
            return symbol in held_symbols or norm in held_symbols
        elif action == "SELL":
            return symbol not in held_symbols and norm not in held_symbols
        return False

    def update_outcomes(self) -> Dict[str, int]:
        """
        Check price outcomes for all signals that have unfilled windows.
        Uses Yahoo Finance for historical prices (free, no rate limit issues).
        """
        updated = {"checked": 0, "filled": 0, "errors": 0}

        # Group signals needing updates by symbol
        needs_update = defaultdict(list)
        today = datetime.now().date()

        for sig in self.signals:
            if sig["action"] == "WATCH":
                continue
            if not sig.get("entry_price"):
                continue
            sig_date = datetime.strptime(sig["date"], "%Y-%m-%d").date()

            for window_str, outcome in sig["outcomes"].items():
                window = int(window_str)
                if outcome.get("return_pct") is not None:
                    continue  # Already filled

                # Check if enough trading days have passed (~1.4x calendar days)
                calendar_days_needed = int(window * 1.45) + 1
                if (today - sig_date).days < calendar_days_needed:
                    continue  # Too early

                needs_update[sig["symbol"]].append((sig, window_str, window))

        if not needs_update:
            print("  No outcomes to update (all filled or too early)")
            return updated

        print(f"  Updating outcomes for {len(needs_update)} symbols...")

        for symbol, entries in needs_update.items():
            try:
                # Determine date range needed
                earliest_date = min(
                    datetime.strptime(e[0]["date"], "%Y-%m-%d").date() for e in entries
                )
                start = earliest_date - timedelta(days=5)
                end = today + timedelta(days=1)

                # Fetch price history via yfinance
                yf_symbol = symbol.replace("/", "-")  # BTC/USD -> BTC-USD
                df = yf.download(yf_symbol, start=start, end=end,
                                 progress=False, auto_adjust=True)

                if df.empty:
                    updated["errors"] += 1
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                close = df['Close']

                for sig, window_str, window in entries:
                    sig_date = datetime.strptime(sig["date"], "%Y-%m-%d").date()
                    entry_price = sig["entry_price"]

                    # Find the trading day that is `window` days after signal
                    trading_days_after = close[close.index.date > sig_date]
                    if len(trading_days_after) >= window:
                        target_price = float(trading_days_after.iloc[window - 1])
                        ret_pct = ((target_price - entry_price) / entry_price) * 100

                        if sig["action"] == "SELL":
                            ret_pct = -ret_pct  # Invert for sell signals

                        sig["outcomes"][window_str] = {
                            "price": round(target_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "checked": today.isoformat(),
                        }
                        updated["filled"] += 1

                    # Update max favorable/adverse excursion
                    all_prices_after = close[close.index.date > sig_date]
                    if len(all_prices_after) > 0 and entry_price:
                        if sig["action"] == "BUY":
                            max_price = float(all_prices_after.max())
                            min_price = float(all_prices_after.min())
                            mfe = ((max_price - entry_price) / entry_price) * 100
                            mae = ((min_price - entry_price) / entry_price) * 100
                        else:
                            max_price = float(all_prices_after.max())
                            min_price = float(all_prices_after.min())
                            mfe = ((entry_price - min_price) / entry_price) * 100
                            mae = ((entry_price - max_price) / entry_price) * 100

                        sig["max_favorable_excursion_pct"] = round(mfe, 2)
                        sig["max_adverse_excursion_pct"] = round(mae, 2)

                        # Check if SL/TP would have been hit
                        sl_pct = sig.get("suggested_sl_pct") or 5.0
                        tp_pct = sig.get("suggested_tp_pct") or 12.0
                        if sig["action"] == "BUY":
                            sig["hit_sl"] = mae <= -sl_pct
                            sig["hit_tp"] = mfe >= tp_pct
                        else:
                            sig["hit_sl"] = mfe <= -sl_pct
                            sig["hit_tp"] = mae >= tp_pct

                    updated["checked"] += 1

                time.sleep(0.2)  # Rate limit courtesy

            except Exception as e:
                print(f"    Error updating {symbol}: {e}")
                updated["errors"] += 1

        self._save_log()
        return updated

    def get_portfolio_pnl_for_signals(self):
        """
        Cross-reference signal log with snapshots to get actual execution P&L.
        Reads from snapshots/ to find position P&L for executed signals.
        """
        filled = 0
        snapshot_dates = []
        if os.path.exists(SNAPSHOTS_DIR):
            snapshot_dates = sorted([
                d for d in os.listdir(SNAPSHOTS_DIR)
                if os.path.isdir(os.path.join(SNAPSHOTS_DIR, d))
            ])

        # Build a map of symbol -> P&L from most recent snapshot
        latest_pnl = {}
        for date_dir in reversed(snapshot_dates):
            dir_path = os.path.join(SNAPSHOTS_DIR, date_dir)
            for fname in os.listdir(dir_path):
                if "post_rebalance" in fname or "snapshot" in fname:
                    fpath = os.path.join(dir_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            snap = json.load(f)
                        for pos in snap.get("positions", []):
                            sym = pos.get("symbol", "")
                            pnl = pos.get("unrealized_plpc", None)
                            if sym and pnl is not None:
                                try:
                                    latest_pnl[sym] = float(pnl)
                                except (ValueError, TypeError):
                                    pass
                    except Exception:
                        pass
            if latest_pnl:
                break  # Use most recent snapshot that has data

        for sig in self.signals:
            if sig.get("was_executed") and sig.get("execution_pnl_pct") is None:
                sym = sig["symbol"]
                norm = sym.replace("/", "")
                pnl = latest_pnl.get(sym) or latest_pnl.get(norm)
                if pnl is not None:
                    sig["execution_pnl_pct"] = round(pnl * 100, 2)
                    filled += 1

        if filled > 0:
            self._save_log()
        return filled


# =============================================================================
# PERFORMANCE ANALYTICS (Feature #1 continued + #4 Weighted Consensus)
# =============================================================================

class PerformanceAnalytics:
    """
    Computes hit rates, win rates, and performance metrics across every
    dimension: by AI engine, sector, regime, confidence tier, technical bias,
    agreement level, and time window.
    
    Also computes the WEIGHTED CONSENSUS parameters for Module 01.
    """

    def __init__(self, signals: List[Dict]):
        self.signals = signals
        self.buy_signals = [s for s in signals if s["action"] == "BUY"]
        self.sell_signals = [s for s in signals if s["action"] == "SELL"]

    def _hit_rate(self, sigs: List[Dict], window: str = "5") -> Optional[float]:
        """% of signals where 5d return > 0 (for BUY) or < 0 (for SELL)."""
        valid = [s for s in sigs if s["outcomes"].get(window, {}).get("return_pct") is not None]
        if len(valid) < MIN_SIGNALS_PER_BUCKET:
            return None
        hits = sum(1 for s in valid if s["outcomes"][window]["return_pct"] > 0)
        return round(hits / len(valid) * 100, 1)

    def _tp_hit_rate(self, sigs: List[Dict]) -> Optional[float]:
        """% of signals where price reached take-profit before stop-loss."""
        valid = [s for s in sigs if s.get("hit_tp") is not None and s.get("hit_sl") is not None]
        if len(valid) < MIN_SIGNALS_PER_BUCKET:
            return None
        # TP hit (regardless of SL) is the "signal was right" metric
        hits = sum(1 for s in valid if s["hit_tp"])
        return round(hits / len(valid) * 100, 1)

    def _avg_return(self, sigs: List[Dict], window: str = "5") -> Optional[float]:
        """Average return at given window."""
        rets = [s["outcomes"][window]["return_pct"] for s in sigs
                if s["outcomes"].get(window, {}).get("return_pct") is not None]
        if len(rets) < MIN_SIGNALS_PER_BUCKET:
            return None
        return round(float(np.mean(rets)), 2)

    def _avg_mfe(self, sigs: List[Dict]) -> Optional[float]:
        """Average max favorable excursion."""
        vals = [s["max_favorable_excursion_pct"] for s in sigs
                if s.get("max_favorable_excursion_pct") is not None]
        if len(vals) < MIN_SIGNALS_PER_BUCKET:
            return None
        return round(float(np.mean(vals)), 2)

    def _avg_mae(self, sigs: List[Dict]) -> Optional[float]:
        """Average max adverse excursion."""
        vals = [s["max_adverse_excursion_pct"] for s in sigs
                if s.get("max_adverse_excursion_pct") is not None]
        if len(vals) < MIN_SIGNALS_PER_BUCKET:
            return None
        return round(float(np.mean(vals)), 2)

    def _bucket_stats(self, sigs: List[Dict], label: str = "") -> Dict:
        """Compute full stats for a bucket of signals."""
        return {
            "label": label,
            "count": len(sigs),
            "hit_rate_1d": self._hit_rate(sigs, "1"),
            "hit_rate_5d": self._hit_rate(sigs, "5"),
            "hit_rate_20d": self._hit_rate(sigs, "20"),
            "tp_hit_rate": self._tp_hit_rate(sigs),
            "avg_return_1d": self._avg_return(sigs, "1"),
            "avg_return_5d": self._avg_return(sigs, "5"),
            "avg_return_20d": self._avg_return(sigs, "20"),
            "avg_mfe": self._avg_mfe(sigs),
            "avg_mae": self._avg_mae(sigs),
        }

    def compute_all_stats(self) -> Dict:
        """Compute performance across every dimension."""
        stats = {
            "generated": datetime.now().isoformat(),
            "total_signals": len(self.signals),
            "total_buy_signals": len(self.buy_signals),
            "total_sell_signals": len(self.sell_signals),
        }

        if len(self.buy_signals) < MIN_SIGNALS_FOR_STATS:
            stats["status"] = "insufficient_data"
            stats["message"] = (
                f"Need at least {MIN_SIGNALS_FOR_STATS} BUY signals with outcomes. "
                f"Currently have {len(self.buy_signals)}. Keep running --log daily."
            )
            return stats

        stats["status"] = "ready"

        # --- OVERALL ---
        stats["overall"] = self._bucket_stats(self.buy_signals, "All BUY signals")

        # --- BY AI ENGINE ---
        claude_full = [s for s in self.buy_signals if s.get("ai_agreement") == "full"]
        claude_only = [s for s in self.buy_signals if s.get("ai_agreement") == "single"
                       and s.get("source_claude_conf", 0) > s.get("source_gpt_conf", 0)]
        gpt_only = [s for s in self.buy_signals if s.get("ai_agreement") == "single"
                    and s.get("source_gpt_conf", 0) >= s.get("source_claude_conf", 0)]

        stats["by_ai_agreement"] = {
            "full_consensus": self._bucket_stats(claude_full, "Both AIs agree"),
            "claude_only": self._bucket_stats(claude_only, "Claude-led signals"),
            "gpt_only": self._bucket_stats(gpt_only, "GPT-led signals"),
        }

        # --- BY SECTOR ---
        sectors = defaultdict(list)
        for s in self.buy_signals:
            sectors[s.get("sector", "Unknown")].append(s)
        stats["by_sector"] = {
            sec: self._bucket_stats(sigs, sec)
            for sec, sigs in sectors.items()
        }

        # --- BY REGIME ---
        regimes = defaultdict(list)
        for s in self.buy_signals:
            regimes[s.get("regime", "unknown")].append(s)
        stats["by_regime"] = {
            reg: self._bucket_stats(sigs, reg)
            for reg, sigs in regimes.items()
        }

        # --- BY CONFIDENCE TIER ---
        conf_buckets = {
            "high_conf_85+": [s for s in self.buy_signals if s.get("confidence", 0) >= 0.85],
            "good_conf_70_84": [s for s in self.buy_signals if 0.70 <= s.get("confidence", 0) < 0.85],
            "moderate_conf_55_69": [s for s in self.buy_signals if 0.55 <= s.get("confidence", 0) < 0.70],
            "low_conf_below_55": [s for s in self.buy_signals if s.get("confidence", 0) < 0.55],
        }
        stats["by_confidence"] = {
            name: self._bucket_stats(sigs, name)
            for name, sigs in conf_buckets.items()
        }

        # --- BY TECHNICAL BIAS ---
        tech_buckets = defaultdict(list)
        for s in self.buy_signals:
            tech_buckets[s.get("technical_bias", "neutral")].append(s)
        stats["by_technical_bias"] = {
            bias: self._bucket_stats(sigs, bias)
            for bias, sigs in tech_buckets.items()
        }

        # --- BY CONVICTION TIER ---
        conviction_buckets = defaultdict(list)
        for s in self.buy_signals:
            conviction_buckets[s.get("conviction_tier", "base")].append(s)
        stats["by_conviction_tier"] = {
            tier: self._bucket_stats(sigs, tier)
            for tier, sigs in conviction_buckets.items()
        }

        # --- COMBINED: full agreement + bullish technicals ---
        gold_standard = [
            s for s in self.buy_signals
            if s.get("ai_agreement") == "full"
            and s.get("technical_bias") in ("bullish", "mildly_bullish")
        ]
        stats["gold_standard_signals"] = self._bucket_stats(
            gold_standard, "Full consensus + bullish technicals"
        )

        # --- WEIGHTED CONSENSUS PARAMETERS ---
        stats["weighted_consensus"] = self._compute_weighted_consensus()

        # --- FAILURE PATTERN ANALYSIS ---
        stats["failure_patterns"] = self._analyze_failure_patterns()

        return stats

    def _compute_weighted_consensus(self) -> Dict:
        """
        Compute dynamic weights for Claude vs GPT based on hit rates.
        Used by Module 01 to weight consensus scoring.
        """
        # Get 5-day hit rates for each AI source
        full_consensus = [s for s in self.buy_signals if s.get("ai_agreement") == "full"]
        claude_led = [s for s in self.buy_signals if s.get("ai_agreement") == "single"
                      and s.get("source_claude_conf", 0) > s.get("source_gpt_conf", 0)]
        gpt_led = [s for s in self.buy_signals if s.get("ai_agreement") == "single"
                   and s.get("source_gpt_conf", 0) >= s.get("source_claude_conf", 0)]

        claude_hr = self._hit_rate(full_consensus + claude_led, "5") or 50.0
        gpt_hr = self._hit_rate(full_consensus + gpt_led, "5") or 50.0
        consensus_hr = self._hit_rate(full_consensus, "5") or 60.0

        total = claude_hr + gpt_hr
        if total == 0:
            total = 100.0

        claude_weight = round(claude_hr / total, 3)
        gpt_weight = round(gpt_hr / total, 3)

        # Consensus bonus: how much extra to add when both agree
        consensus_bonus = max(0, consensus_hr - max(claude_hr, gpt_hr))

        return {
            "claude_weight": claude_weight,
            "gpt_weight": gpt_weight,
            "claude_hit_rate_5d": claude_hr,
            "gpt_hit_rate_5d": gpt_hr,
            "consensus_hit_rate_5d": consensus_hr,
            "consensus_bonus_pct": round(consensus_bonus, 1),
            "recommendation": (
                f"Weight Claude at {claude_weight:.0%}, GPT at {gpt_weight:.0%}. "
                f"Full consensus signals get +{consensus_bonus:.0f}% bonus. "
                f"Based on {len(self.buy_signals)} historical signals."
            ),
        }

    def _analyze_failure_patterns(self) -> List[Dict]:
        """
        Find the most common patterns among LOSING signals.
        Returns top failure modes for the AI to learn from.
        """
        # Signals where 5d return was negative
        losers = [
            s for s in self.buy_signals
            if s["outcomes"].get("5", {}).get("return_pct") is not None
            and s["outcomes"]["5"]["return_pct"] < 0
        ]

        if len(losers) < 3:
            return []

        patterns = []

        # Pattern: Buying against trend (below SMA200)
        below_sma200 = [s for s in losers if s.get("above_sma200") is False]
        if below_sma200:
            patterns.append({
                "pattern": "Buying below SMA200",
                "count": len(below_sma200),
                "pct_of_losses": round(len(below_sma200) / len(losers) * 100, 1),
                "avg_loss": round(np.mean([s["outcomes"]["5"]["return_pct"] for s in below_sma200]), 2),
            })

        # Pattern: Buying into weak sectors
        sector_loss_rate = defaultdict(lambda: {"losses": 0, "total": 0})
        for s in self.buy_signals:
            if s["outcomes"].get("5", {}).get("return_pct") is not None:
                sec = s.get("sector", "Unknown")
                sector_loss_rate[sec]["total"] += 1
                if s["outcomes"]["5"]["return_pct"] < 0:
                    sector_loss_rate[sec]["losses"] += 1

        worst_sectors = []
        for sec, data in sector_loss_rate.items():
            if data["total"] >= 3:
                loss_rate = data["losses"] / data["total"] * 100
                if loss_rate > 60:
                    worst_sectors.append({"sector": sec, "loss_rate": round(loss_rate, 1),
                                          "sample_size": data["total"]})

        if worst_sectors:
            patterns.append({
                "pattern": "Weak sector selections",
                "sectors": sorted(worst_sectors, key=lambda x: x["loss_rate"], reverse=True),
            })

        # Pattern: Low confidence signals that failed
        low_conf_losers = [s for s in losers if s.get("confidence", 0) < 0.70]
        if low_conf_losers:
            patterns.append({
                "pattern": "Low confidence signals (<70%) that failed",
                "count": len(low_conf_losers),
                "avg_loss": round(np.mean([s["outcomes"]["5"]["return_pct"] for s in low_conf_losers]), 2),
                "suggestion": "Consider raising minimum confidence threshold",
            })

        # Pattern: Single-AI disagreement signals that failed
        single_losers = [s for s in losers if s.get("ai_agreement") == "single"]
        if single_losers:
            patterns.append({
                "pattern": "Single-AI signals (no consensus) that failed",
                "count": len(single_losers),
                "pct_of_losses": round(len(single_losers) / len(losers) * 100, 1),
            })

        # Pattern: Bearish technicals + BUY signal
        bearish_buys = [s for s in losers if s.get("technical_bias") in ("bearish", "mildly_bearish")]
        if bearish_buys:
            patterns.append({
                "pattern": "BUY signal with bearish technicals",
                "count": len(bearish_buys),
                "avg_loss": round(np.mean([s["outcomes"]["5"]["return_pct"] for s in bearish_buys]), 2),
                "suggestion": "Do not override bearish technicals with AI conviction alone",
            })

        return patterns


# =============================================================================
# REGIME CONSTRAINTS GENERATOR (Feature #5)
# =============================================================================

class RegimeConstraintsGenerator:
    """
    Generates regime-specific trading rules based on actual signal performance
    data per regime. These rules are injected into AI prompts as non-negotiable
    constraints, creating a data-driven feedback loop.
    """

    # Default constraints when insufficient data
    DEFAULT_CONSTRAINTS = {
        "bull": {
            "max_confidence_cap": 1.0,
            "min_reward_risk": 2.0,
            "preferred_setups": ["momentum breakouts", "pullbacks to support", "sector leaders"],
            "avoid_setups": [],
            "sector_overweight": [],
            "sector_underweight": [],
            "notes": "Bull regime: be aggressive, ride momentum, let winners run.",
        },
        "sideways": {
            "max_confidence_cap": 0.90,
            "min_reward_risk": 2.5,
            "preferred_setups": ["range-bound mean reversion", "dividend/income plays", "low-beta sectors"],
            "avoid_setups": ["momentum breakouts (high failure rate in sideways)"],
            "sector_overweight": [],
            "sector_underweight": [],
            "notes": "Sideways regime: be selective, demand catalysts, tighter sizing.",
        },
        "bear": {
            "max_confidence_cap": 0.80,
            "min_reward_risk": 3.0,
            "preferred_setups": ["oversold bounces with clear catalysts", "defensive sectors", "short setups"],
            "avoid_setups": ["momentum chasing", "speculative growth", "high-beta names"],
            "sector_overweight": ["Utilities", "Healthcare", "Consumer Staples"],
            "sector_underweight": ["Technology", "Consumer Discretionary"],
            "notes": "Bear regime: be defensive, smaller positions, tighter stops, more cash.",
        },
    }

    def __init__(self, stats: Dict):
        self.stats = stats

    def generate(self) -> Dict:
        """Generate data-driven regime constraints."""
        constraints = {}

        # Start with defaults
        for regime in ["bull", "sideways", "bear"]:
            constraints[regime] = dict(self.DEFAULT_CONSTRAINTS.get(regime, {}))

        # Overlay with actual data if available
        regime_stats = self.stats.get("by_regime", {})
        sector_stats = self.stats.get("by_sector", {})

        for regime_name, regime_data in regime_stats.items():
            # Normalize regime name
            normalized = "bull" if "bull" in regime_name.lower() else \
                         "bear" if "bear" in regime_name.lower() else "sideways"

            if normalized not in constraints:
                constraints[normalized] = dict(self.DEFAULT_CONSTRAINTS.get("sideways", {}))

            c = constraints[normalized]

            # Adjust confidence cap based on actual hit rate
            hr = regime_data.get("hit_rate_5d")
            if hr is not None:
                if hr < 45:
                    c["max_confidence_cap"] = 0.75
                    c["notes"] += f" Data shows only {hr}% hit rate — be very selective."
                elif hr < 55:
                    c["max_confidence_cap"] = 0.85
                elif hr > 70:
                    c["max_confidence_cap"] = 1.0
                    c["notes"] += f" Strong {hr}% hit rate — confidence well calibrated."

            # Adjust R:R based on avg MFE/MAE
            mfe = regime_data.get("avg_mfe")
            mae = regime_data.get("avg_mae")
            if mfe is not None and mae is not None and mae != 0:
                actual_rr = abs(mfe / mae) if mae else 1.0
                c["actual_reward_risk"] = round(actual_rr, 2)
                if actual_rr < 1.5:
                    c["min_reward_risk"] = 3.0
                    c["notes"] += " Low actual R:R — tighten entries."

        # Identify best/worst sectors from data
        if sector_stats:
            ranked = []
            for sec, data in sector_stats.items():
                hr = data.get("hit_rate_5d")
                if hr is not None:
                    ranked.append((sec, hr, data.get("count", 0)))

            ranked.sort(key=lambda x: x[1], reverse=True)

            if ranked:
                best = [s[0] for s in ranked[:3] if s[1] > 55]
                worst = [s[0] for s in ranked[-3:] if s[1] < 45]

                for regime in constraints.values():
                    if best:
                        regime["sector_overweight"] = best
                    if worst:
                        regime["sector_underweight"] = worst

        # Add failure patterns as "avoid" rules
        failure_patterns = self.stats.get("failure_patterns", [])
        for fp in failure_patterns:
            pattern = fp.get("pattern", "")
            for regime in constraints.values():
                if pattern not in regime.get("avoid_setups", []):
                    regime.setdefault("avoid_setups", []).append(pattern)

        result = {
            "generated": datetime.now().isoformat(),
            "data_backed": self.stats.get("status") == "ready",
            "constraints": constraints,
        }

        save_json(REGIME_CONSTRAINTS_FILE, result)
        return result


# =============================================================================
# PATTERN LIBRARY (Feature #6)
# =============================================================================

class PatternLibrary:
    """
    Extracts winning technical signature patterns from signal history.
    
    Identifies combinations of indicators that historically led to
    profitable trades, ranked by win rate and sample size.
    """

    def __init__(self, signals: List[Dict]):
        self.buy_signals = [
            s for s in signals
            if s["action"] == "BUY"
            and s["outcomes"].get("5", {}).get("return_pct") is not None
        ]

    def build(self) -> Dict:
        """Build pattern library from signal history."""
        if len(self.buy_signals) < MIN_SIGNALS_FOR_STATS:
            return {
                "generated": datetime.now().isoformat(),
                "status": "insufficient_data",
                "patterns": [],
            }

        patterns = []

        # Define pattern templates
        pattern_defs = [
            {
                "name": "Trend Momentum (above SMA50+200, bullish technicals, strong momentum)",
                "filter": lambda s: (
                    s.get("above_sma50") is True
                    and s.get("above_sma200") is True
                    and s.get("technical_bias") in ("bullish", "mildly_bullish")
                    and (s.get("momentum_1m") or 0) > 3
                ),
            },
            {
                "name": "RSI Power Zone (RSI 40-60, above SMA200, volume surge)",
                "filter": lambda s: (
                    s.get("rsi") is not None
                    and 40 <= (s.get("rsi") or 0) <= 60
                    and s.get("above_sma200") is True
                    and (s.get("volume_ratio") or 0) > 1.3
                ),
            },
            {
                "name": "High Conviction Consensus (85%+ conf, full agreement, bullish tech)",
                "filter": lambda s: (
                    s.get("confidence", 0) >= 0.85
                    and s.get("ai_agreement") == "full"
                    and s.get("technical_bias") in ("bullish", "mildly_bullish")
                ),
            },
            {
                "name": "Sector Leader (RS vs SPY > 5%, above both SMAs)",
                "filter": lambda s: (
                    (s.get("rs_vs_spy") or 0) > 5
                    and s.get("above_sma50") is True
                    and s.get("above_sma200") is True
                ),
            },
            {
                "name": "Oversold Bounce (RSI < 35, above SMA200)",
                "filter": lambda s: (
                    s.get("rsi") is not None
                    and (s.get("rsi") or 100) < 35
                    and s.get("above_sma200") is True
                ),
            },
            {
                "name": "Breakout Setup (above SMA50, volume > 1.5x, positive momentum)",
                "filter": lambda s: (
                    s.get("above_sma50") is True
                    and (s.get("volume_ratio") or 0) > 1.5
                    and (s.get("momentum_1m") or 0) > 0
                ),
            },
            {
                "name": "Defensive Quality (low conf but strong technicals + full consensus)",
                "filter": lambda s: (
                    s.get("confidence", 0) < 0.75
                    and s.get("ai_agreement") == "full"
                    and s.get("technical_bias") in ("bullish",)
                ),
            },
            {
                "name": "Contrarian Recovery (below SMA50, bullish divergence, high volume)",
                "filter": lambda s: (
                    s.get("above_sma50") is False
                    and s.get("above_sma200") is True
                    and (s.get("volume_ratio") or 0) > 1.2
                    and s.get("technical_bias") in ("bullish", "mildly_bullish")
                ),
            },
        ]

        for pdef in pattern_defs:
            matching = [s for s in self.buy_signals if pdef["filter"](s)]
            if len(matching) < 3:
                continue

            winners = [s for s in matching if s["outcomes"]["5"]["return_pct"] > 0]
            win_rate = round(len(winners) / len(matching) * 100, 1)

            avg_ret = round(np.mean([s["outcomes"]["5"]["return_pct"] for s in matching]), 2)
            avg_mfe = np.mean([s["max_favorable_excursion_pct"] for s in matching
                               if s.get("max_favorable_excursion_pct") is not None])
            avg_mfe = round(avg_mfe, 2) if not np.isnan(avg_mfe) else None

            patterns.append({
                "name": pdef["name"],
                "sample_size": len(matching),
                "win_rate_5d": win_rate,
                "avg_return_5d": avg_ret,
                "avg_mfe": avg_mfe,
                "quality": "A" if win_rate > 65 and len(matching) >= 5 else
                           "B" if win_rate > 55 else "C",
            })

        patterns.sort(key=lambda x: (x["quality"], -x["win_rate_5d"]))

        result = {
            "generated": datetime.now().isoformat(),
            "status": "ready" if patterns else "insufficient_data",
            "total_signals_analyzed": len(self.buy_signals),
            "patterns": patterns,
        }

        save_json(PATTERN_LIBRARY_FILE, result)
        return result


# =============================================================================
# POST-MORTEM AGENT (Feature #3)
# =============================================================================

class PostMortemAgent:
    """
    Uses Claude (with extended thinking) to analyze recent closed trades
    and generate actionable insights.
    """

    def __init__(self, signals: List[Dict], stats: Dict):
        self.signals = signals
        self.stats = stats
        self.cfg = Config()
        self._claude = None

    @property
    def claude(self):
        if self._claude is None and self.cfg.ANTHROPIC_API_KEY:
            self._claude = anthropic.Anthropic(api_key=self.cfg.ANTHROPIC_API_KEY)
        return self._claude

    def run_weekly_review(self, lookback_days: int = 7) -> Optional[Dict]:
        """Run AI post-mortem on recent signals with outcomes."""
        if not self.claude:
            print("  Anthropic API key not configured — skipping post-mortem")
            return None

        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        recent = [
            s for s in self.signals
            if s["date"] >= cutoff
            and s["action"] == "BUY"
            and s["outcomes"].get("5", {}).get("return_pct") is not None
        ]

        if len(recent) < 3:
            print(f"  Only {len(recent)} signals with 5d outcomes in last {lookback_days} days — need 3+")
            return None

        # Prepare trade data for AI
        winners = [s for s in recent if s["outcomes"]["5"]["return_pct"] > 0]
        losers = [s for s in recent if s["outcomes"]["5"]["return_pct"] <= 0]

        def fmt_signal(s):
            return {
                "symbol": s["symbol"],
                "sector": s.get("sector"),
                "confidence": s.get("confidence"),
                "agreement": s.get("ai_agreement"),
                "technical_bias": s.get("technical_bias"),
                "regime": s.get("regime"),
                "entry_price": s.get("entry_price"),
                "return_5d": s["outcomes"]["5"]["return_pct"],
                "max_favorable": s.get("max_favorable_excursion_pct"),
                "max_adverse": s.get("max_adverse_excursion_pct"),
                "hit_tp": s.get("hit_tp"),
                "hit_sl": s.get("hit_sl"),
                "rsi": s.get("rsi"),
                "above_sma50": s.get("above_sma50"),
                "above_sma200": s.get("above_sma200"),
            }

        # Build the analysis prompt
        prompt = f"""You are the Chief Risk Officer of Luxverum Capital reviewing the past week's trading signals.

SIGNAL PERFORMANCE (last {lookback_days} days):
- Total signals analyzed: {len(recent)}
- Winners (5d return > 0): {len(winners)} ({len(winners)/len(recent)*100:.0f}%)
- Losers (5d return <= 0): {len(losers)} ({len(losers)/len(recent)*100:.0f}%)

WINNING TRADES:
{json.dumps([fmt_signal(s) for s in winners], indent=2)}

LOSING TRADES:
{json.dumps([fmt_signal(s) for s in losers], indent=2)}

CURRENT PERFORMANCE STATS:
{json.dumps({k: v for k, v in self.stats.items() if k in ('overall', 'failure_patterns', 'weighted_consensus')}, indent=2)}

Analyze these results and provide:

1. LOSING TRADE AUTOPSY: For each losing trade, what did the signal miss? Was there a better indicator? Should regime context have prevented entry?

2. WINNER OPTIMIZATION: Did we capture enough of the move? Compare 5d return vs max_favorable_excursion. Should trailing stops be tighter or looser?

3. PATTERN INSIGHTS: What distinguishes winners from losers in this sample? (sector, technicals, agreement level, confidence, regime alignment)

4. SPECIFIC PARAMETER ADJUSTMENTS: Give exactly 3 concrete recommendations with:
   - What to change (e.g., "raise minimum confidence to 70% in sideways regimes")
   - Expected impact (e.g., "would have filtered 4 of 6 losers")
   - Trade-off (e.g., "would have also filtered 1 winner")

5. PROMPT ENGINEERING SUGGESTIONS: What specific instructions should we add to the AI discovery prompts to avoid the failure patterns you identified?

Be specific, quantitative, and actionable. Reference actual symbols and numbers."""

        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 6000,
                },
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response (may include thinking blocks)
            analysis_text = ""
            for block in response.content:
                if block.type == "text":
                    analysis_text += block.text

            result = {
                "generated": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "signals_analyzed": len(recent),
                "win_rate": round(len(winners) / len(recent) * 100, 1),
                "analysis": analysis_text,
                "model_used": "claude-sonnet-4-20250514",
                "used_extended_thinking": True,
            }

            save_json("postmortem_latest.json", result)
            return result

        except Exception as e:
            print(f"  Post-mortem API error: {e}")
            traceback.print_exc()
            return None


# =============================================================================
# FEEDBACK CONTEXT BUILDER (ties it all together for Module 01)
# =============================================================================

class FeedbackContextBuilder:
    """
    Generates the ai_feedback_context.json file that Module 01 injects
    into Claude and GPT prompts. This is the key integration point.
    """

    def __init__(self, stats: Dict, constraints: Dict, patterns: Dict,
                 postmortem: Optional[Dict] = None):
        self.stats = stats
        self.constraints = constraints
        self.patterns = patterns
        self.postmortem = postmortem

    def build(self) -> Dict:
        """Build the complete feedback context for AI prompts."""
        context = {
            "generated": datetime.now().isoformat(),
            "version": "1.0",
        }

        # Section 1: Performance Report Card
        context["performance_report"] = self._build_performance_section()

        # Section 2: Regime-specific constraints (formatted as prompt text)
        context["regime_prompt_injection"] = self._build_regime_prompt()

        # Section 3: Pattern library summary
        context["pattern_prompt_injection"] = self._build_pattern_prompt()

        # Section 4: Weighted consensus parameters
        wc = self.stats.get("weighted_consensus", {})
        context["consensus_weights"] = {
            "claude_weight": wc.get("claude_weight", 0.5),
            "gpt_weight": wc.get("gpt_weight", 0.5),
            "consensus_bonus_pct": wc.get("consensus_bonus_pct", 10),
        }

        # Section 5: Post-mortem insights (if available)
        if self.postmortem and self.postmortem.get("analysis"):
            # Extract just the key recommendations, keep it concise for prompt injection
            context["postmortem_insights"] = self._extract_postmortem_key_points()

        save_json(FEEDBACK_CONTEXT_FILE, context)
        return context

    def _build_performance_section(self) -> str:
        """Build human-readable performance report for AI prompts."""
        if self.stats.get("status") != "ready":
            return "SIGNAL PERFORMANCE: Insufficient data (collecting signals, check back in 1-2 weeks)."

        overall = self.stats.get("overall", {})
        wc = self.stats.get("weighted_consensus", {})

        lines = [
            "HISTORICAL SIGNAL PERFORMANCE (from our tracking system):",
            f"  Total signals tracked: {self.stats.get('total_buy_signals', 0)}",
        ]

        if overall.get("hit_rate_5d") is not None:
            lines.append(f"  Overall 5-day hit rate: {overall['hit_rate_5d']}%")
        if overall.get("tp_hit_rate") is not None:
            lines.append(f"  Take-profit hit rate: {overall['tp_hit_rate']}%")
        if overall.get("avg_return_5d") is not None:
            lines.append(f"  Average 5-day return: {overall['avg_return_5d']:+.2f}%")
        if overall.get("avg_mfe") is not None:
            lines.append(f"  Average max favorable excursion: {overall['avg_mfe']:+.2f}%")
        if overall.get("avg_mae") is not None:
            lines.append(f"  Average max adverse excursion: {overall['avg_mae']:+.2f}%")

        # AI-specific performance
        by_ai = self.stats.get("by_ai_agreement", {})
        for key, label in [("full_consensus", "Full consensus"),
                           ("claude_only", "Claude-led"),
                           ("gpt_only", "GPT-led")]:
            data = by_ai.get(key, {})
            hr = data.get("hit_rate_5d")
            if hr is not None:
                lines.append(f"  {label} hit rate (5d): {hr}% (n={data.get('count', 0)})")

        # Best/worst sectors
        sector_data = self.stats.get("by_sector", {})
        if sector_data:
            ranked = [(sec, d.get("hit_rate_5d"), d.get("count", 0))
                      for sec, d in sector_data.items()
                      if d.get("hit_rate_5d") is not None]
            ranked.sort(key=lambda x: x[1], reverse=True)

            if ranked:
                best = ranked[0]
                worst = ranked[-1]
                lines.append(f"  Best sector: {best[0]} ({best[1]}% hit rate, n={best[2]})")
                lines.append(f"  Worst sector: {worst[0]} ({worst[1]}% hit rate, n={worst[2]})")

        # Confidence calibration
        conf_data = self.stats.get("by_confidence", {})
        for key in ["high_conf_85+", "good_conf_70_84"]:
            d = conf_data.get(key, {})
            hr = d.get("hit_rate_5d")
            if hr is not None:
                lines.append(f"  {key.replace('_', ' ').title()}: {hr}% hit rate")

        # Failure patterns
        failures = self.stats.get("failure_patterns", [])
        if failures:
            lines.append("")
            lines.append("  KNOWN FAILURE PATTERNS (avoid these):")
            for fp in failures[:4]:
                pattern = fp.get("pattern", "")
                count = fp.get("count", "")
                suggestion = fp.get("suggestion", "")
                lines.append(f"    - {pattern}" +
                             (f" ({count} instances)" if count else "") +
                             (f" -> {suggestion}" if suggestion else ""))

        return "\n".join(lines)

    def _build_regime_prompt(self) -> Dict[str, str]:
        """Build per-regime prompt text that Module 01 can inject."""
        result = {}
        constraints_data = self.constraints.get("constraints", {})

        for regime, rules in constraints_data.items():
            lines = [
                f"REGIME TRADING RULES FOR {regime.upper()} MARKET (data-backed, non-negotiable):",
            ]

            cap = rules.get("max_confidence_cap", 1.0)
            if cap < 1.0:
                lines.append(f"  - Maximum confidence for any signal: {cap:.0%}")

            rr = rules.get("min_reward_risk", 2.0)
            lines.append(f"  - Minimum reward-to-risk ratio: {rr:.1f}:1")

            preferred = rules.get("preferred_setups", [])
            if preferred:
                lines.append(f"  - PREFERRED setups: {', '.join(preferred)}")

            avoid = rules.get("avoid_setups", [])
            if avoid:
                lines.append(f"  - AVOID these setups: {', '.join(avoid)}")

            overweight = rules.get("sector_overweight", [])
            if overweight:
                lines.append(f"  - Overweight sectors (historically strong): {', '.join(overweight)}")

            underweight = rules.get("sector_underweight", [])
            if underweight:
                lines.append(f"  - Underweight sectors (historically weak): {', '.join(underweight)}")

            actual_rr = rules.get("actual_reward_risk")
            if actual_rr:
                lines.append(f"  - Our actual R:R in this regime: {actual_rr:.2f}:1")

            notes = rules.get("notes", "")
            if notes:
                lines.append(f"  NOTE: {notes}")

            result[regime] = "\n".join(lines)

        return result

    def _build_pattern_prompt(self) -> str:
        """Build pattern library summary for AI prompts."""
        patterns = self.patterns.get("patterns", [])
        if not patterns:
            return "PATTERN LIBRARY: Still building (insufficient signal history)."

        lines = [
            "HIGH-PROBABILITY TECHNICAL PATTERNS (from our signal tracking):",
            "Score each opportunity against these patterns and flag matches.",
            "",
        ]

        for i, p in enumerate(patterns[:6], 1):
            quality = p.get("quality", "?")
            name = p.get("name", "Unknown")
            wr = p.get("win_rate_5d", 0)
            n = p.get("sample_size", 0)
            avg_ret = p.get("avg_return_5d", 0)

            lines.append(
                f"  Pattern {i} [{quality}]: {name}"
            )
            lines.append(
                f"    Win rate: {wr}% | Avg return: {avg_ret:+.2f}% | Sample: {n} signals"
            )

        return "\n".join(lines)

    def _extract_postmortem_key_points(self) -> str:
        """Extract key recommendations from post-mortem analysis."""
        analysis = self.postmortem.get("analysis", "")
        if not analysis:
            return ""

        # Take the last portion which typically contains recommendations
        # Truncate to keep prompt injection reasonable
        if len(analysis) > 1500:
            # Try to find the "PARAMETER ADJUSTMENTS" or "RECOMMENDATIONS" section
            for marker in ["PARAMETER ADJUSTMENTS", "SPECIFIC PARAMETER", "RECOMMENDATIONS",
                           "PROMPT ENGINEERING", "SUGGESTIONS"]:
                idx = analysis.upper().find(marker)
                if idx > 0:
                    analysis = analysis[idx:]
                    break
            else:
                analysis = analysis[-1500:]

        return f"WEEKLY POST-MORTEM INSIGHTS (AI-generated):\n{analysis[:1500]}"


# =============================================================================
# REPORT PRINTER
# =============================================================================

def print_report(stats: Dict, constraints: Dict, patterns: Dict):
    """Print human-readable performance report to console."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  LUXVERUM CAPITAL — SIGNAL INTELLIGENCE REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)

    if stats.get("status") != "ready":
        print(f"\n  {stats.get('message', 'Insufficient data')}")
        print(f"  Total signals logged: {stats.get('total_signals', 0)}")
        print(f"\n  Keep running --log daily. Stats unlock at {MIN_SIGNALS_FOR_STATS}+ signals.")
        return

    # Overall
    overall = stats.get("overall", {})
    print(f"\n  OVERALL BUY SIGNAL PERFORMANCE (n={overall.get('count', 0)})")
    print(f"  {'─' * 50}")
    for window in ["1d", "5d", "20d"]:
        hr = overall.get(f"hit_rate_{window}")
        ar = overall.get(f"avg_return_{window}")
        if hr is not None:
            print(f"    {window:>3} hit rate: {hr:>5.1f}%   avg return: {ar:>+6.2f}%")

    tp_hr = overall.get("tp_hit_rate")
    if tp_hr is not None:
        print(f"    TP hit rate: {tp_hr:.1f}%")

    mfe = overall.get("avg_mfe")
    mae = overall.get("avg_mae")
    if mfe is not None:
        print(f"    Avg MFE: {mfe:+.2f}%  |  Avg MAE: {mae:+.2f}%")

    # AI Agreement
    print(f"\n  BY AI AGREEMENT")
    print(f"  {'─' * 50}")
    for key, label in [("full_consensus", "Both AIs agree"),
                       ("claude_only", "Claude-led"),
                       ("gpt_only", "GPT-led")]:
        data = stats.get("by_ai_agreement", {}).get(key, {})
        hr = data.get("hit_rate_5d")
        n = data.get("count", 0)
        if hr is not None:
            print(f"    {label:20s}  5d HR: {hr:>5.1f}%  (n={n})")

    # Weighted consensus
    wc = stats.get("weighted_consensus", {})
    if wc:
        print(f"\n  WEIGHTED CONSENSUS ENGINE")
        print(f"  {'─' * 50}")
        print(f"    Claude weight: {wc.get('claude_weight', 0.5):.0%}")
        print(f"    GPT weight:    {wc.get('gpt_weight', 0.5):.0%}")
        print(f"    Consensus bonus: +{wc.get('consensus_bonus_pct', 0):.0f}%")

    # Sectors
    print(f"\n  BY SECTOR")
    print(f"  {'─' * 50}")
    sector_data = stats.get("by_sector", {})
    ranked = [(sec, d.get("hit_rate_5d"), d.get("count", 0), d.get("avg_return_5d"))
              for sec, d in sector_data.items()
              if d.get("hit_rate_5d") is not None]
    ranked.sort(key=lambda x: x[1], reverse=True)
    for sec, hr, n, ar in ranked:
        bar = "#" * int(hr / 5) if hr else ""
        print(f"    {sec:22s}  {hr:>5.1f}%  avg:{ar:>+5.2f}%  n={n:>2}  {bar}")

    # Patterns
    patterns_list = patterns.get("patterns", [])
    if patterns_list:
        print(f"\n  PATTERN LIBRARY")
        print(f"  {'─' * 50}")
        for p in patterns_list[:6]:
            q = p.get("quality", "?")
            name = p.get("name", "")[:50]
            wr = p.get("win_rate_5d", 0)
            print(f"    [{q}] {name:50s} WR: {wr:.1f}%")

    # Failure patterns
    failures = stats.get("failure_patterns", [])
    if failures:
        print(f"\n  FAILURE PATTERNS (learn from these)")
        print(f"  {'─' * 50}")
        for fp in failures[:4]:
            print(f"    ! {fp.get('pattern', 'Unknown')}")
            if fp.get("suggestion"):
                print(f"      -> {fp['suggestion']}")

    print(f"\n{sep}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Module 9: Signal Intelligence Engine"
    )
    parser.add_argument("--log", action="store_true",
                        help="Log today's signals + update past outcomes")
    parser.add_argument("--stats", action="store_true",
                        help="Compute performance statistics")
    parser.add_argument("--feedback", action="store_true",
                        help="Generate AI feedback context for Module 01")
    parser.add_argument("--postmortem", action="store_true",
                        help="Run AI post-mortem on recent trades")
    parser.add_argument("--patterns", action="store_true",
                        help="Build/update pattern library")
    parser.add_argument("--full", action="store_true",
                        help="Run everything: log + stats + feedback + patterns")
    parser.add_argument("--report", action="store_true",
                        help="Print human-readable intelligence report")
    parser.add_argument("--postmortem-days", type=int, default=7,
                        help="Lookback days for post-mortem (default: 7)")

    args = parser.parse_args()

    # Default to --full if no flags
    if not any([args.log, args.stats, args.feedback, args.postmortem,
                args.patterns, args.full, args.report]):
        args.full = True

    print_header("SIGNAL INTELLIGENCE ENGINE")

    # ── STEP 1: LOG ──
    if args.log or args.full:
        print("\n  Step 1: Logging today's signals...")
        tracker = SignalTracker()
        logged = tracker.log_todays_signals()
        print(f"    New signals logged: {logged}")
        print(f"    Total signals in database: {len(tracker.signals)}")

        print("\n  Step 2: Updating historical outcomes...")
        updates = tracker.update_outcomes()
        print(f"    Outcomes filled: {updates.get('filled', 0)}")
        print(f"    Errors: {updates.get('errors', 0)}")

        print("\n  Step 3: Cross-referencing portfolio P&L...")
        pnl_filled = tracker.get_portfolio_pnl_for_signals()
        print(f"    P&L entries matched: {pnl_filled}")

        signals = tracker.signals
    else:
        # Load existing log
        tracker = SignalTracker()
        signals = tracker.signals

    # ── STEP 2: STATS ──
    stats = {}
    if args.stats or args.feedback or args.full or args.report:
        print("\n  Computing performance analytics...")
        analytics = PerformanceAnalytics(signals)
        stats = analytics.compute_all_stats()
        save_json(SIGNAL_PERF_FILE, stats)
        print(f"    Status: {stats.get('status', 'unknown')}")
        if stats.get("status") == "ready":
            overall = stats.get("overall", {})
            print(f"    Overall 5d hit rate: {overall.get('hit_rate_5d', 'N/A')}%")
            wc = stats.get("weighted_consensus", {})
            print(f"    Claude weight: {wc.get('claude_weight', 0.5):.0%} | "
                  f"GPT weight: {wc.get('gpt_weight', 0.5):.0%}")

    # ── STEP 3: REGIME CONSTRAINTS ──
    constraints = {}
    if args.feedback or args.full:
        print("\n  Generating regime constraints...")
        gen = RegimeConstraintsGenerator(stats)
        constraints = gen.generate()
        data_backed = constraints.get("data_backed", False)
        print(f"    Data-backed: {data_backed}")

    # ── STEP 4: PATTERNS ──
    patterns = {}
    if args.patterns or args.full:
        print("\n  Building pattern library...")
        lib = PatternLibrary(signals)
        patterns = lib.build()
        n_patterns = len(patterns.get("patterns", []))
        print(f"    Patterns discovered: {n_patterns}")

    # ── STEP 5: POST-MORTEM ──
    postmortem = None
    if args.postmortem or args.full:
        print(f"\n  Running AI post-mortem (lookback: {args.postmortem_days}d)...")
        agent = PostMortemAgent(signals, stats)
        postmortem = agent.run_weekly_review(lookback_days=args.postmortem_days)
        if postmortem:
            print(f"    Analysis generated ({len(postmortem.get('analysis', ''))} chars)")
            print(f"    Win rate this period: {postmortem.get('win_rate', 'N/A')}%")
        else:
            print("    Skipped (insufficient data or API unavailable)")

    # ── STEP 6: BUILD FEEDBACK CONTEXT ──
    if args.feedback or args.full:
        print("\n  Building feedback context for Module 01...")
        # Load patterns if not already built
        if not patterns:
            patterns = load_json(PATTERN_LIBRARY_FILE) or {}
        builder = FeedbackContextBuilder(stats, constraints, patterns, postmortem)
        context = builder.build()
        print(f"    Feedback context saved to data/{FEEDBACK_CONTEXT_FILE}")
        print(f"    Performance report: {len(context.get('performance_report', '').split(chr(10)))} lines")
        regime_prompts = context.get("regime_prompt_injection", {})
        print(f"    Regime prompts: {len(regime_prompts)} regimes configured")

    # ── REPORT ──
    if args.report or args.full:
        if not stats:
            stats = load_json(SIGNAL_PERF_FILE) or {}
        if not constraints:
            constraints = load_json(REGIME_CONSTRAINTS_FILE) or {}
        if not patterns:
            patterns = load_json(PATTERN_LIBRARY_FILE) or {}
        print_report(stats, constraints, patterns)

    print(f"\n  {'=' * 50}")
    print(f"  Signal Intelligence Engine complete.")
    print(f"  {'=' * 50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())