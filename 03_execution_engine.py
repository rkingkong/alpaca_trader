#!/usr/bin/env python3
"""
Module 3: Unified Execution Engine
====================================
Consolidates portfolio rebalancing, trade execution, protection management,
and order lifecycle into a single coherent module.

Capabilities:
  1. Score all positions + new opportunities -> unified ranking
  2. Calculate target portfolio allocation (regime-adaptive & volatility-adjusted)
  3. Execute trades with SMART BRACKET protection (limit entry + dynamic stops)
  4. Audit and repair unprotected positions
  5. Pre/post trade snapshots for reconciliation
  6. Sector concentration enforcement (max 30% per sector)
  7. Max drawdown circuit breaker (halt new buys if drawdown > 15%)
  8. Market hours awareness for equity orders

Usage:
  python 03_execution_engine.py --rebalance              # Full rebalance
  python 03_execution_engine.py --rebalance --dry-run     # Preview only
  python 03_execution_engine.py --new-trades              # Only add new positions
  python 03_execution_engine.py --protect-only            # Fix unprotected positions
  python 03_execution_engine.py --graduate                # Graduate winners to trailing stops
  python 03_execution_engine.py --graduate --dry-run      # Preview graduations
  python 03_execution_engine.py --audit                   # Read-only protection check
"""

import os
import sys
import json
import time
import re
import argparse
import logging
from datetime import datetime, date, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple, Set
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict

import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
    StopLimitOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockBarsRequest,
    CryptoBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

# Shared config
try:
    from config import (
        Config, Clients, load_regime_context, RegimeContext,
        DATA_DIR, LOGS_DIR, SNAPSHOTS_DIR,
        load_json, save_json, print_header, print_step,
        setup_logging, is_market_hours, is_extended_hours, market_status,
        update_drawdown, load_drawdown_state,
        MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT,
        MAX_SECTOR_CONCENTRATION_PCT, MAX_SINGLE_POSITION_PCT,
        SYMBOL_SECTOR_MAP, get_sector,
        recommendations_age_seconds, MAX_RECOMMENDATIONS_AGE_SECONDS,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        Config, Clients, load_regime_context, RegimeContext,
        DATA_DIR, LOGS_DIR, SNAPSHOTS_DIR,
        load_json, save_json, print_header, print_step,
        setup_logging, is_market_hours, is_extended_hours, market_status,
        update_drawdown, load_drawdown_state,
        MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT,
        MAX_SECTOR_CONCENTRATION_PCT, MAX_SINGLE_POSITION_PCT,
        SYMBOL_SECTOR_MAP, get_sector,
        recommendations_age_seconds, MAX_RECOMMENDATIONS_AGE_SECONDS,
    )

logger = setup_logging("execution_engine")


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Unified execution engine for all trading operations.

    Workflows:
      rebalance()     -> Score everything, compute target, execute delta
      new_trades()    -> Only add new positions from AI recommendations
      protect_only()  -> Just fix unprotected positions
      audit()         -> Read-only protection check
    """

    # --- DEFAULT PARAMETERS (overridden by regime context) --------------------

    # Position sizing (conviction-based)
    MAX_POSITION_PCT = 7.0          # High conviction
    MED_POSITION_PCT = 4.5          # Medium conviction
    MIN_POSITION_PCT = 2.5          # Base conviction
    HIGH_CONVICTION_THRESHOLD = 0.72
    MED_CONVICTION_THRESHOLD = 0.58
    MIN_CONFIDENCE_THRESHOLD = 0.45

    # Portfolio limits
    MAX_POSITIONS = 20
    CASH_RESERVE_PCT = 3.0
    TRIM_THRESHOLD_PCT = 12.0       # Trim if position exceeds this

    # Risk management
    STOP_LOSS_PCT = 7.0
    TAKE_PROFIT_PCT = 20.0
    TRAILING_STOP_TRAIL_PCT = 4.0
    TRAILING_STOP_TRIGGER_GAIN_PCT = 5.0
    TRAILING_TIGHTEN_GAIN_PCT = 15.0
    TRAILING_TIGHTEN_TRAIL_PCT = 3.0
    DAILY_LOSS_LIMIT_PCT = 4.0
    POSITION_HARD_STOP_PCT = 15.0   # Absolute max loss per position

    # Scoring weights (must sum to 100)
    W_AI_CONFIDENCE = 30
    W_TECHNICAL = 20
    W_MOMENTUM = 25
    W_PNL = 15
    W_TREND = 10

    # Rebalance thresholds
    SELL_SCORE_THRESHOLD = 30       # Sell if score < 30

    # Order lifecycle
    ORDER_FILL_TIMEOUT_SEC = 60
    ORDER_POLL_INTERVAL_SEC = 2
    CANCEL_TIMEOUT_SEC = 15

    # Crypto symbol normalization (Alpaca returns BTC/USD for orders, BTCUSD for positions)
    _CRYPTO_PAIRS = {
        "BTCUSD": "BTC/USD", "BTC/USD": "BTCUSD",
        "ETHUSD": "ETH/USD", "ETH/USD": "ETHUSD",
        "SOLUSD": "SOL/USD", "SOL/USD": "SOLUSD",
        "AVAXUSD": "AVAX/USD", "AVAX/USD": "AVAXUSD",
        "DOGEUSD": "DOGE/USD", "DOGE/USD": "DOGEUSD",
        "LTCUSD": "LTC/USD", "LTC/USD": "LTCUSD",
        "LINKUSD": "LINK/USD", "LINK/USD": "LINKUSD",
        "UNIUSD": "UNI/USD", "UNI/USD": "UNIUSD",
        "SHIBUSD": "SHIB/USD", "SHIB/USD": "SHIBUSD",
    }

    @staticmethod
    def _normalize_crypto_symbol(symbol: str) -> str:
        """Convert crypto symbol to slash format (BTC/USD) for order submission."""
        if '/' in symbol:
            return symbol
        if symbol in ExecutionEngine._CRYPTO_PAIRS:
            return ExecutionEngine._CRYPTO_PAIRS[symbol]
        if symbol.endswith("USD") and len(symbol) > 3:
            return symbol[:-3] + "/" + "USD"
        return symbol

    @staticmethod
    def _crypto_symbol_variants(symbol: str) -> set:
        """Return all symbol variants for a crypto symbol."""
        variants = {symbol}
        if '/' in symbol:
            variants.add(symbol.replace("/", ""))
        elif symbol.endswith("USD") and len(symbol) > 3:
            variants.add(symbol[:-3] + "/USD")
        if symbol in ExecutionEngine._CRYPTO_PAIRS:
            variants.add(ExecutionEngine._CRYPTO_PAIRS[symbol])
        return variants

    @staticmethod
    def _is_crypto_symbol(symbol: str) -> bool:
        """Check if a symbol is crypto."""
        if '/' in symbol:
            return True
        if symbol in ExecutionEngine._CRYPTO_PAIRS:
            return True
        if symbol.endswith("USD") and len(symbol) > 3:
            base = symbol[:-3]
            return base in ("BTC", "ETH", "SOL", "AVAX", "DOGE", "LTC", "LINK", "UNI", "SHIB")
        return False

    @staticmethod
    def _is_option_symbol(symbol: str) -> bool:
        """Check if symbol is an OCC-format option contract."""
        if not symbol or len(symbol) < 10:
            return False
        return bool(re.match(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$', symbol))

    def __init__(self):
        cfg = Config()
        self.paper_trading = cfg.PAPER_TRADING

        self.trading_client = TradingClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, paper=cfg.PAPER_TRADING
        )
        self.stock_data = StockHistoricalDataClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY
        )
        self.crypto_data = CryptoHistoricalDataClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY
        )

        self.log = {
            "timestamp": datetime.now().isoformat(),
            "regime": None,
            "actions": [],
            "errors": [],
            "warnings": [],
            "protection_audit": [],
        }

        self.regime: Optional[RegimeContext] = None
        self._recently_protected: set = set()
        self._adapt_to_regime()

    # --- REGIME ADAPTATION ----------------------------------------------------

    def _adapt_to_regime(self):
        """Load regime context and adapt all parameters."""
        ctx = load_regime_context()
        if ctx is None:
            logger.warning("No fresh regime context -- using defaults")
            return

        self.regime = ctx
        self.log["regime"] = ctx.regime

        self.STOP_LOSS_PCT = ctx.recommended_stop_loss_pct
        self.TAKE_PROFIT_PCT = ctx.recommended_take_profit_pct
        self.MIN_CONFIDENCE_THRESHOLD = ctx.recommended_confidence_threshold
        self.MAX_POSITIONS = ctx.recommended_max_positions
        self.CASH_RESERVE_PCT = ctx.recommended_cash_reserve_pct

        r = ctx.risk_appetite
        self.MAX_POSITION_PCT = round(3.0 + r * 5.0, 1)
        self.MED_POSITION_PCT = round(2.0 + r * 3.5, 1)
        self.MIN_POSITION_PCT = round(1.0 + r * 2.0, 1)

        self.TRIM_THRESHOLD_PCT = round(self.MAX_POSITION_PCT * 1.5, 1)

        regime_str = ctx.regime.upper().replace('_', ' ')
        logger.info(f"Regime: {regime_str} (score: {ctx.regime_score:+.3f})")
        logger.info(f"  Risk: {r:.0%} | SL: {self.STOP_LOSS_PCT}% | TP: {self.TAKE_PROFIT_PCT}%")
        logger.info(f"  Positions: {self.MIN_POSITION_PCT}-{self.MAX_POSITION_PCT}% | Max: {self.MAX_POSITIONS}")

        if 'bull' in ctx.regime:
            self.TRAILING_STOP_TRAIL_PCT = 5.0
            self.TRAILING_STOP_TRIGGER_GAIN_PCT = 5.0
            self.TRAILING_TIGHTEN_TRAIL_PCT = 3.5
        elif 'bear' in ctx.regime:
            self.TRAILING_STOP_TRAIL_PCT = 3.0
            self.TRAILING_STOP_TRIGGER_GAIN_PCT = 3.0
            self.TRAILING_TIGHTEN_TRAIL_PCT = 2.0
        else:
            self.TRAILING_STOP_TRAIL_PCT = 4.0
            self.TRAILING_STOP_TRIGGER_GAIN_PCT = 5.0
            self.TRAILING_TIGHTEN_TRAIL_PCT = 3.0
        logger.info(
            f"  Trail: {self.TRAILING_STOP_TRAIL_PCT}% | "
            f"Trigger: +{self.TRAILING_STOP_TRIGGER_GAIN_PCT}% | "
            f"Tighten: +{self.TRAILING_TIGHTEN_GAIN_PCT}%"
        )

    # --- PRE-TRADE VALIDATION -------------------------------------------------

    def _validate_pre_trade(self, account: Dict, dry_run: bool) -> List[str]:
        """
        Run institutional pre-trade checks. Returns list of blocking issues.
        Empty list = safe to proceed.
        """
        issues = []

        # 1. Daily loss circuit breaker
        if account["daily_pnl_pct"] <= -self.DAILY_LOSS_LIMIT_PCT:
            issues.append(
                f"CIRCUIT BREAKER: Daily loss {account['daily_pnl_pct']:.1f}% "
                f"exceeds {self.DAILY_LOSS_LIMIT_PCT}% limit"
            )

        # 2. Max drawdown circuit breaker
        dd_state = update_drawdown(account["equity"])
        if dd_state.current_drawdown_pct >= MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT:
            issues.append(
                f"DRAWDOWN BREAKER: Current drawdown {dd_state.current_drawdown_pct:.1f}% "
                f"from peak ${dd_state.high_water_mark:,.0f} "
                f"exceeds {MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT}% limit"
            )

        # 3. Trading blocked
        if account.get("trading_blocked"):
            issues.append("ACCOUNT BLOCKED: Alpaca reports trading is blocked")

        # 4. Stale recommendations guard
        age = recommendations_age_seconds()
        if age is not None and age > MAX_RECOMMENDATIONS_AGE_SECONDS:
            issues.append(
                f"STALE SIGNALS: Recommendations are {age / 3600:.1f}h old "
                f"(max {MAX_RECOMMENDATIONS_AGE_SECONDS / 3600:.0f}h)"
            )

        for issue in issues:
            logger.warning(f"  {issue}")
            self.log["warnings"].append(issue)

        return issues

    def _compute_sector_exposure(self, positions: Dict, equity: float) -> Dict[str, float]:
        """
        Calculate current portfolio sector exposure as percentages.
        Returns {sector: pct_of_equity}.
        """
        exposure = defaultdict(float)
        for sym, pos in positions.items():
            if pos.get("is_option"):
                continue
            value = abs(pos.get("market_value", 0))
            sector = get_sector(sym)
            exposure[sector] += value
        # Convert to percentages
        return {sector: round(val / equity * 100, 2) for sector, val in exposure.items()} if equity > 0 else {}

    def _would_breach_sector_limit(self, symbol: str, buy_value: float,
                                    positions: Dict, equity: float) -> bool:
        """Check if buying this symbol would breach sector concentration limits."""
        sector = get_sector(symbol)
        if sector == "Unknown":
            return False  # Can't check, allow it

        exposure = self._compute_sector_exposure(positions, equity)
        current_sector_pct = exposure.get(sector, 0.0)
        additional_pct = (buy_value / equity * 100) if equity > 0 else 0

        if current_sector_pct + additional_pct > MAX_SECTOR_CONCENTRATION_PCT:
            logger.warning(
                f"  SECTOR LIMIT: {symbol} ({sector}) would push sector to "
                f"{current_sector_pct + additional_pct:.1f}% "
                f"(limit: {MAX_SECTOR_CONCENTRATION_PCT}%)"
            )
            return True
        return False

    # --- DATA FETCHING --------------------------------------------------------

    def get_account(self) -> Dict:
        acct = self.trading_client.get_account()
        equity = float(acct.equity)
        last_eq = float(acct.last_equity)
        return {
            "equity": equity,
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "last_equity": last_eq,
            "daily_pnl": round(equity - last_eq, 2),
            "daily_pnl_pct": round(((equity / last_eq) - 1) * 100, 2) if last_eq > 0 else 0,
            "pattern_day_trader": acct.pattern_day_trader,
            "trading_blocked": acct.trading_blocked,
        }

    def get_positions(self) -> Dict[str, Dict]:
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            sym = pos.symbol
            is_option = self._is_option_symbol(sym) or (
                hasattr(pos.asset_class, 'value') and pos.asset_class.value == "us_option"
            )
            result[sym] = {
                "symbol": sym,
                "qty": float(pos.qty),
                "qty_available": float(pos.qty_available) if pos.qty_available else float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                "asset_class": pos.asset_class.value if hasattr(pos.asset_class, 'value') else str(pos.asset_class),
                "is_option": is_option,
            }
        return result

    def get_open_orders(self) -> List[Dict]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        orders = self.trading_client.get_orders(filter=request)
        return [self._order_to_dict(o) for o in orders]

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            if self._is_crypto_symbol(symbol):
                from alpaca.data.requests import CryptoLatestQuoteRequest
                query_sym = self._normalize_crypto_symbol(symbol)
                req = CryptoLatestQuoteRequest(symbol_or_symbols=query_sym)
                quotes = self.crypto_data.get_crypto_latest_quote(req)
            else:
                query_sym = symbol
                req = StockLatestQuoteRequest(symbol_or_symbols=query_sym)
                quotes = self.stock_data.get_stock_latest_quote(req)

            if query_sym in quotes:
                q = quotes[query_sym]
                bid = float(q.bid_price) if q.bid_price else 0
                ask = float(q.ask_price) if q.ask_price else 0
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2, 4)
                return bid or ask or None
        except Exception:
            pass
        return None

    def _find_recommendation(self, symbol: str, recommendations: Dict) -> Dict:
        rec = recommendations.get(symbol)
        if rec:
            return rec
        if self._is_crypto_symbol(symbol):
            for variant in self._crypto_symbol_variants(symbol):
                rec = recommendations.get(variant)
                if rec:
                    return rec
        return {}

    # --- SCORING ENGINE -------------------------------------------------------

    def score_position(self, symbol: str, position: Dict, recommendations: Dict) -> float:
        """Score an existing position using the 5-factor model."""
        rec = self._find_recommendation(symbol, recommendations)
        score = 0.0

        # Factor 1: AI Confidence (30%)
        confidence = rec.get("confidence", 0.0)
        action = rec.get("action", "").upper()
        if action == "SELL":
            score += 0
        elif action == "BUY":
            score += confidence * self.W_AI_CONFIDENCE
        else:
            score += confidence * self.W_AI_CONFIDENCE * 0.5

        # Factor 2: Technical Bias (20%)
        tech_bias = rec.get("technical_bias", "neutral").lower()
        tech_scores = {"bullish": 1.0, "mildly_bullish": 0.7, "neutral": 0.4,
                       "mildly_bearish": 0.2, "bearish": 0.0}
        score += tech_scores.get(tech_bias, 0.4) * self.W_TECHNICAL

        # Factor 3: Momentum / P&L Performance (25%)
        pnl_pct = position.get("unrealized_plpc", 0)
        if pnl_pct > 15:
            score += self.W_MOMENTUM * 1.0
        elif pnl_pct > 5:
            score += self.W_MOMENTUM * 0.8
        elif pnl_pct > 0:
            score += self.W_MOMENTUM * 0.6
        elif pnl_pct > -5:
            score += self.W_MOMENTUM * 0.3
        elif pnl_pct > -10:
            score += self.W_MOMENTUM * 0.15
        else:
            score += 0

        # Factor 4: P&L Contribution (15%)
        if pnl_pct > 0:
            score += min(self.W_PNL, pnl_pct * 0.5)
        else:
            score += max(0, self.W_PNL + pnl_pct * 0.5)

        # Factor 5: Trend Alignment (10%)
        if self.regime:
            if "bull" in self.regime.regime:
                if pnl_pct > 0 and action != "SELL":
                    score += self.W_TREND
                else:
                    score += self.W_TREND * 0.3
            elif "bear" in self.regime.regime:
                score += self.W_TREND * 0.2
            else:
                score += self.W_TREND * 0.5

        return round(min(100, max(0, score)), 1)

    def score_opportunity(self, rec: Dict) -> float:
        """Score a new buy opportunity."""
        score = 0.0
        confidence = rec.get("confidence", 0)
        action = rec.get("action", "").upper()

        if action != "BUY":
            return 0.0

        score += confidence * self.W_AI_CONFIDENCE

        tech_bias = rec.get("technical_bias", "neutral").lower()
        tech_scores = {"bullish": 1.0, "mildly_bullish": 0.7, "neutral": 0.4,
                       "mildly_bearish": 0.2, "bearish": 0.0}
        score += tech_scores.get(tech_bias, 0.4) * self.W_TECHNICAL

        conviction = rec.get("conviction_tier", "base").lower()
        conviction_scores = {"high": 1.0, "medium": 0.65, "base": 0.35}
        score += conviction_scores.get(conviction, 0.35) * self.W_MOMENTUM

        agreement = rec.get("ai_agreement", "").lower()
        if "full" in agreement:
            score += 12
        elif "partial" in agreement:
            score += 5

        if self.regime and "bull" in self.regime.regime:
            if confidence >= 0.70:
                score += 5

        return round(min(100, max(0, score)), 1)

    # --- TARGET ALLOCATION (Risk-Parity + Sector-Aware) -----------------------

    def _size_for_score(self, score: float, atr_pct: float = None) -> float:
        """
        Risk-Parity Sizing: adjusts capital allocation based on volatility (ATR).
        Higher-volatility positions get smaller allocations to equalize risk.
        """
        if score >= 80:
            base_pct = self.MAX_POSITION_PCT
        elif score >= 60:
            base_pct = self.MED_POSITION_PCT
        else:
            base_pct = self.MIN_POSITION_PCT

        if not atr_pct or atr_pct <= 0:
            return base_pct

        # Baseline assumption: 2.5% daily ATR is "normal"
        vol_factor = 2.5 / atr_pct

        # Clamp between 0.4x and 1.5x (no extreme leverage or over-reduction)
        vol_factor = max(0.4, min(vol_factor, 1.5))

        target_pct = base_pct * vol_factor
        return round(min(target_pct, self.MAX_POSITION_PCT * 1.5), 1)

    def compute_target_portfolio(self, account: Dict, positions: Dict,
                                  recommendations: Dict) -> List[Dict]:
        """
        Compute the ideal target portfolio allocation.
        Returns list of {symbol, action, target_pct, current_pct, score, ...}
        """
        equity = account["equity"]
        targets = []

        # Score existing positions
        for sym, pos in positions.items():
            if pos.get("is_option"):
                continue

            current_pct = (abs(pos["market_value"]) / equity * 100) if equity > 0 else 0
            score = self.score_position(sym, pos, recommendations)

            rec = self._find_recommendation(sym, recommendations)
            atr_pct = rec.get("indicators", {}).get("atr_percent", 2.5) if rec else 2.5

            # Penalize oversized positions
            if current_pct > self.MAX_POSITION_PCT * 1.3:
                score = round(score * 0.8, 1)

            if score < self.SELL_SCORE_THRESHOLD:
                action = "SELL"
                target_pct = 0.0
            elif current_pct > self.TRIM_THRESHOLD_PCT:
                action = "TRIM"
                target_pct = self._size_for_score(score, atr_pct)
            elif score >= 70 and current_pct < self.MED_POSITION_PCT:
                action = "ADD"
                target_pct = self._size_for_score(score, atr_pct)
            else:
                action = "HOLD"
                target_pct = current_pct

            targets.append({
                "symbol": sym,
                "action": action,
                "score": score,
                "current_pct": round(current_pct, 2),
                "target_pct": round(target_pct, 2),
                "pnl_pct": round(pos.get("unrealized_plpc", 0), 1),
                "is_existing": True,
                "atr_pct": atr_pct,
                "sector": get_sector(sym),
            })

        # Identify held symbols (including all crypto variants)
        held_symbols = set()
        for sym in positions.keys():
            held_symbols.add(sym)
            if self._is_crypto_symbol(sym):
                held_symbols.update(self._crypto_symbol_variants(sym))

        # Score new buy opportunities
        buy_recs = [r for r in recommendations.values()
                    if r.get("action", "").upper() == "BUY"
                    and r.get("symbol") not in held_symbols
                    and r.get("confidence", 0) >= self.MIN_CONFIDENCE_THRESHOLD]

        for rec in buy_recs:
            score = self.score_opportunity(rec)
            if score >= 40:
                atr_pct = rec.get("indicators", {}).get("atr_percent", 2.5)
                target_pct = self._size_for_score(score, atr_pct)
                targets.append({
                    "symbol": rec["symbol"],
                    "action": "BUY",
                    "score": score,
                    "current_pct": 0.0,
                    "target_pct": round(target_pct, 2),
                    "confidence": rec.get("confidence", 0),
                    "sector": rec.get("sector", get_sector(rec["symbol"])),
                    "is_existing": False,
                    "atr_pct": atr_pct,
                })

        targets.sort(key=lambda x: x["score"], reverse=True)

        buys = [t for t in targets if t["action"] == "BUY"]
        holds_and_adds = [t for t in targets if t["action"] in ("HOLD", "ADD")]
        sells_and_trims = [t for t in targets if t["action"] in ("SELL", "TRIM")]

        # Force-sell weakest if over max positions
        if len(holds_and_adds) > self.MAX_POSITIONS:
            holds_and_adds.sort(key=lambda x: x["score"])
            excess = len(holds_and_adds) - self.MAX_POSITIONS
            for t in holds_and_adds[:excess]:
                t["action"] = "SELL"
                t["target_pct"] = 0.0
                sells_and_trims.append(t)
            holds_and_adds = [t for t in holds_and_adds if t["action"] in ("HOLD", "ADD")]
            logger.warning(f"Force-selling {excess} weakest positions to meet max limit of {self.MAX_POSITIONS}")

        available_slots = self.MAX_POSITIONS - len(holds_and_adds)
        accepted_buys = buys[:max(0, available_slots)]

        # Enforce sector concentration on new buys
        sector_checked_buys = []
        sector_exposure = self._compute_sector_exposure(positions, account["equity"])
        for buy in accepted_buys:
            sector = buy.get("sector", "Unknown")
            current_sector_pct = sector_exposure.get(sector, 0.0)
            additional_pct = buy["target_pct"]
            if sector != "Unknown" and current_sector_pct + additional_pct > MAX_SECTOR_CONCENTRATION_PCT:
                logger.info(
                    f"  Skipping {buy['symbol']}: {sector} sector at "
                    f"{current_sector_pct:.1f}% + {additional_pct:.1f}% "
                    f"would exceed {MAX_SECTOR_CONCENTRATION_PCT}% limit"
                )
                continue
            sector_checked_buys.append(buy)
            sector_exposure[sector] = current_sector_pct + additional_pct

        accepted_buys = sector_checked_buys

        # Scale if total exceeds investable amount
        active = holds_and_adds + accepted_buys
        total_target = sum(t["target_pct"] for t in active)
        investable = 100.0 - self.CASH_RESERVE_PCT
        if total_target > investable:
            scale = investable / total_target
            for t in active:
                t["target_pct"] = round(t["target_pct"] * scale, 2)

        return sells_and_trims + active

    def _value_to_shares(self, value: float, price: float, symbol: str) -> float:
        """Convert dollar value to share quantity."""
        if price <= 0:
            return 0
        if self._is_crypto_symbol(symbol):
            return round(value / price, 6)
        else:
            return max(0, int(value / price))

    # --- TRADE EXECUTION ------------------------------------------------------

    def execute_plan(self, targets: List[Dict], account: Dict,
                     positions: Dict, dry_run: bool = True) -> List[Dict]:
        """Execute the target portfolio plan. Returns list of executed actions."""
        equity = account["equity"]
        executed = []

        # Pre-trade validation
        issues = self._validate_pre_trade(account, dry_run)
        if issues and not dry_run:
            logger.error("Pre-trade validation FAILED -- halting new buys")
            # Still allow sells to reduce exposure, but block buys
            targets = [t for t in targets if t["action"] in ("SELL", "TRIM")]

        # Market hours check for equities
        mkt = market_status()
        can_trade_equities = mkt in ("OPEN", "EXTENDED_HOURS")
        if not can_trade_equities and not dry_run:
            logger.info(f"Market is {mkt}. Equity orders will use GTC time-in-force.")

        # Phase 1: SELLS and TRIMS first (free up capital)
        sells = [t for t in targets if t["action"] in ("SELL", "TRIM")]
        for target in sells:
            sym = target["symbol"]
            pos = positions.get(sym)
            if not pos:
                continue

            if target["action"] == "SELL":
                qty = pos["qty"]
                label = "SELL ALL"
            else:
                target_value = equity * target["target_pct"] / 100
                current_value = abs(pos["market_value"])
                reduce_value = current_value - target_value
                if reduce_value <= 0:
                    continue
                qty = self._value_to_shares(reduce_value, pos["current_price"], sym)
                if qty <= 0:
                    continue
                label = "TRIM"

            action = {
                "symbol": sym, "action": label, "side": "sell",
                "qty": qty, "score": target["score"],
                "reason": f"Score {target['score']:.0f} | P&L {target.get('pnl_pct', 0):+.1f}%",
            }

            if dry_run:
                action["status"] = "DRY_RUN"
                print(f"  [DRY] {label} {qty} {sym} (score: {target['score']:.0f})")
            else:
                self._cancel_orders_for_symbol(sym)
                result = self._submit_market_order(sym, qty, OrderSide.SELL)
                action["status"] = "SUBMITTED" if result else "FAILED"
                action["order_id"] = result

            executed.append(action)

        if not dry_run and sells:
            time.sleep(3)

        # Phase 2: BUYS and ADDS
        # IMPORTANT: ADD orders use simple limit buy + unified OCO for full position.
        # Only NEW BUY orders use bracket (since there's no existing protection to unify).
        buys = [t for t in targets if t["action"] in ("BUY", "ADD")]
        for target in buys:
            sym = target["symbol"]
            price = self.get_current_price(sym)
            if not price:
                self.log["warnings"].append(f"No price for {sym} -- skipping")
                continue

            target_value = equity * target["target_pct"] / 100
            current_value = abs(positions.get(sym, {}).get("market_value", 0))
            buy_value = target_value - current_value
            if buy_value < 50:
                continue

            qty = self._value_to_shares(buy_value, price, sym)
            if qty <= 0:
                continue

            if buy_value > account.get("buying_power", 0) * 0.95:
                self.log["warnings"].append(f"Insufficient buying power for {sym}")
                continue

            # Sector concentration guard (final check at execution time)
            if not dry_run and self._would_breach_sector_limit(sym, buy_value, positions, equity):
                self.log["warnings"].append(f"{sym}: skipped (sector concentration limit)")
                continue

            # Dynamic stops based on ATR volatility
            atr_pct = target.get("atr_pct", 0)
            if atr_pct and atr_pct > 0:
                sl_pct = min(max(round(atr_pct * 2.0, 2), 3.0), self.POSITION_HARD_STOP_PCT)
                tp_pct = max(round(atr_pct * 5.0, 2), self.TAKE_PROFIT_PCT)
            else:
                sl_pct = self.STOP_LOSS_PCT
                tp_pct = self.TAKE_PROFIT_PCT

            action = {
                "symbol": sym, "action": target["action"], "side": "buy",
                "qty": qty, "price": price, "value": round(buy_value, 2),
                "score": target["score"],
                "stop_loss_pct": sl_pct,
                "take_profit_pct": tp_pct,
                "sector": target.get("sector", get_sector(sym)),
            }

            is_add = target["action"] == "ADD"

            if dry_run:
                action["status"] = "DRY_RUN"
                sl_price = round(price * (1 - sl_pct / 100), 2)
                tp_price = round(price * (1 + tp_pct / 100), 2)
                mode = "ADD (unified OCO)" if is_add else "BUY (bracket)"
                print(
                    f"  [DRY] {mode} {qty} {sym} @ ${price:.2f} "
                    f"(RiskParity: {target['target_pct']}% | "
                    f"SL: ${sl_price} [-{sl_pct}%] | TP: ${tp_price} [+{tp_pct}%])"
                )
            elif is_add:
                # ADD strategy: cancel existing protection, buy simple, then protect ALL shares
                logger.info(f"  ADD {qty} {sym}: canceling existing protection for unified OCO...")
                self._cancel_orders_for_symbol(sym)
                time.sleep(2)
                result = self._submit_market_order(sym, qty, OrderSide.BUY)
                if result:
                    # Wait for fill, then protect the ENTIRE position
                    time.sleep(5)
                    try:
                        fresh_pos = self.get_positions().get(sym)
                        if fresh_pos:
                            logger.info(f"  Protecting entire {sym} position ({fresh_pos['qty']} shares)...")
                            self._add_oco_protection(sym, fresh_pos)
                        else:
                            logger.warning(f"  {sym}: position not found after ADD, will fix in audit pass")
                    except Exception as e:
                        logger.warning(f"  {sym}: post-ADD protection failed: {e} (will fix in audit)")
                action["status"] = "SUBMITTED" if result else "FAILED"
                action["order_id"] = result
            else:
                # NEW BUY: use bracket order (creates stop+TP atomically with the buy)
                result = self._submit_bracket_order(sym, qty, price, sl_pct, tp_pct)
                action["status"] = "SUBMITTED" if result else "FAILED"
                action["order_id"] = result

            executed.append(action)

        self.log["actions"] = executed
        return executed

    def _cancel_orders_for_symbol(self, symbol: str):
        """Cancel all open sell orders for a symbol (including crypto variants)."""
        try:
            orders = self.get_open_orders()
            sym_variants = (
                self._crypto_symbol_variants(symbol) if self._is_crypto_symbol(symbol)
                else {symbol}
            )
            canceled = 0
            for o in orders:
                if o["symbol"] in sym_variants and o["side"] == "sell":
                    try:
                        self.trading_client.cancel_order_by_id(o["id"])
                        canceled += 1
                    except Exception:
                        pass
            if canceled > 0:
                logger.info(f"  Canceled {canceled} existing order(s) for {symbol}")
                time.sleep(1)
        except Exception:
            pass

    def _submit_market_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[str]:
        """Submit a market order. Uses GTC for sells to prevent overnight exposure gaps."""
        try:
            is_crypto = self._is_crypto_symbol(symbol)
            order_symbol = self._normalize_crypto_symbol(symbol) if is_crypto else symbol
            if is_crypto:
                qty = float(qty)
            else:
                qty = int(float(qty))

            if qty <= 0:
                logger.warning(f"{symbol}: qty is 0 after rounding, skipping")
                return None

            # Use GTC for all sell orders to prevent expiration leaving positions unprotected
            if side == OrderSide.SELL:
                tif = TimeInForce.GTC
            elif is_crypto:
                tif = TimeInForce.GTC
            else:
                tif = TimeInForce.DAY

            request = MarketOrderRequest(
                symbol=order_symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
            )

            order = self.trading_client.submit_order(request)
            logger.info(f"  {side.value.upper()} {qty} {order_symbol} (TIF={tif.value}) -> Order {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"  {side.value.upper()} {qty} {symbol} failed: {e}")
            self.log["errors"].append(f"{side.value} {symbol}: {e}")
            return None

    def _submit_bracket_order(self, symbol: str, qty: float,
                               price: float, sl_pct: float = None, tp_pct: float = None) -> Optional[str]:
        """
        Smart Limit Bracket Order: reduces slippage using limit entry + dynamic stops.
        Falls back to market order + standalone protection on failure.
        """
        try:
            is_crypto = self._is_crypto_symbol(symbol)
            order_symbol = self._normalize_crypto_symbol(symbol) if is_crypto else symbol

            _sl = sl_pct if sl_pct is not None else self.STOP_LOSS_PCT
            _tp = tp_pct if tp_pct is not None else self.TAKE_PROFIT_PCT

            sl_price = round(price * (1 - _sl / 100), 2)
            tp_price = round(price * (1 + _tp / 100), 2)

            # Slight buffer above market for immediate fill
            limit_buy_price = round(price * 1.002, 2) if not is_crypto else round(price * 1.005, 4)

            request = LimitOrderRequest(
                symbol=order_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC if is_crypto else TimeInForce.DAY,
                limit_price=limit_buy_price,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=sl_price),
                take_profit=TakeProfitRequest(limit_price=tp_price),
            )

            order = self.trading_client.submit_order(request)
            logger.info(
                f"  BRACKET BUY {qty} {symbol} @ limit ~${limit_buy_price:.2f} "
                f"(SL: -{_sl:.1f}% | TP: +{_tp:.1f}%) -> Order {order.id}"
            )
            return str(order.id)

        except Exception as e:
            logger.warning(f"  Bracket order failed for {symbol}: {e}")
            logger.info(f"  Falling back to market order + standalone protection...")
            result = self._submit_market_order(symbol, qty, OrderSide.BUY)
            if result:
                time.sleep(3)
                # Add standalone protection for BOTH crypto AND stocks
                try:
                    is_crypto = self._is_crypto_symbol(symbol)
                    safe_qty = round(qty * 0.99, 6) if is_crypto else int(qty)
                    pos_data = {
                        "qty": safe_qty, "qty_available": safe_qty,
                        "avg_entry_price": price, "current_price": price,
                        "asset_class": "crypto" if is_crypto else "us_equity",
                    }
                    if is_crypto:
                        self._add_crypto_protection(symbol, pos_data)
                    else:
                        self._add_oco_protection(symbol, pos_data)
                except Exception as prot_err:
                    self.log["warnings"].append(
                        f"{symbol}: protection after bracket fallback failed: {prot_err}"
                    )
                self.log["warnings"].append(
                    f"{symbol}: bracket failed, used market order + standalone protection"
                )
            return result

    # --- PROTECTION AUDIT & REPAIR --------------------------------------------

    def _build_orders_by_symbol(self, orders: List[Dict]) -> Dict[str, List[Dict]]:
        """Index open orders by symbol (including crypto variants and legs)."""
        result = defaultdict(list)
        for o in orders:
            sym = o["symbol"]
            result[sym].append(o)
            for variant in self._crypto_symbol_variants(sym):
                if variant != sym:
                    result[variant].append(o)
            for leg in o.get("legs", []):
                leg_sym = leg["symbol"]
                result[leg_sym].append(leg)
                for variant in self._crypto_symbol_variants(leg_sym):
                    if variant != leg_sym:
                        result[variant].append(leg)
        return result

    def audit_protection(self, positions: Dict = None,
                          fix: bool = False, dry_run: bool = True) -> Dict:
        """Audit all positions for protection status and optionally fix gaps."""
        if positions is None:
            positions = self.get_positions()

        orders = self.get_open_orders()
        orders_by_symbol = self._build_orders_by_symbol(orders)

        protected = []
        degraded = []
        unprotected = []
        fixed = []

        for sym, pos in positions.items():
            if pos.get("is_option"):
                continue

            sym_orders = orders_by_symbol.get(sym, [])
            protection = self._classify_protection(sym, pos, sym_orders)

            if protection["has_protection"]:
                if protection.get("issues"):
                    degraded.append(protection)
                    _issues_str = ", ".join(protection["issues"])
                    if fix and not dry_run:
                        logger.info(f"  {sym}: degraded ({_issues_str}) -- replacing...")
                        for o in sym_orders:
                            if o.get("side") == "sell":
                                try:
                                    self.trading_client.cancel_order_by_id(o["id"])
                                except Exception:
                                    pass
                        time.sleep(3)
                        # Re-fetch position so qty_available reflects the cancellations
                        fresh_positions = self.get_positions()
                        fresh_pos = fresh_positions.get(sym, pos)
                        result = self._add_oco_protection(sym, fresh_pos)
                        fixed.append({
                            "symbol": sym,
                            "status": "UPGRADED" if result else "FAILED",
                            "issues": protection["issues"],
                        })
                    elif fix and dry_run:
                        print(f"  [DRY] {sym}: would fix degraded ({_issues_str})")
                        fixed.append({"symbol": sym, "status": "DRY_RUN", "issues": protection["issues"]})
                else:
                    protected.append(protection)
            else:
                unprotected.append(protection)
                if fix:
                    if dry_run:
                        sl_price = round(pos["avg_entry_price"] * (1 - self.STOP_LOSS_PCT / 100), 2)
                        tp_price = round(pos["avg_entry_price"] * (1 + self.TAKE_PROFIT_PCT / 100), 2)
                        print(f"  [DRY] Would add OCO for {sym}: SL ${sl_price} / TP ${tp_price}")
                        fixed.append({"symbol": sym, "status": "DRY_RUN"})
                    else:
                        result = self._add_oco_protection(sym, pos)
                        fixed.append({"symbol": sym, "status": "FIXED" if result else "FAILED"})

        options_skipped = sum(1 for p in positions.values() if p.get("is_option"))
        audited_count = len(protected) + len(degraded) + len(unprotected)

        # Successfully fixed positions should count as protected, not degraded
        fixed_symbols = {f["symbol"] for f in fixed if f["status"] in ("FIXED", "UPGRADED")}
        remaining_degraded = [d for d in degraded if d["symbol"] not in fixed_symbols]
        effective_protected = len(protected) + len(fixed_symbols)

        summary = {
            "total_positions": audited_count,
            "options_excluded": options_skipped,
            "protected": effective_protected,
            "degraded": len(remaining_degraded),
            "unprotected": len(unprotected),
            "fixed": len([f for f in fixed if f["status"] in ("FIXED", "DRY_RUN", "UPGRADED")]),
            "details": {
                "protected": [p["symbol"] for p in protected] + list(fixed_symbols),
                "degraded": [
                    {"symbol": d["symbol"], "issues": d.get("issues", []),
                     "stop_price": d.get("stop_price"),
                     "stop_distance_pct": d.get("stop_distance_pct", 0),
                     "stop_qty": d.get("stop_qty", 0), "qty": d.get("qty", 0)}
                    for d in remaining_degraded
                ],
                "unprotected": [p["symbol"] for p in unprotected],
                "fixed": fixed,
            }
        }

        self.log["protection_audit"] = summary
        return summary

    def _classify_protection(self, symbol: str, position: Dict,
                              orders: List[Dict]) -> Dict:
        """Classify the protection status of a position based on its open orders."""
        if symbol in self._recently_protected:
            return {
                "symbol": symbol,
                "has_protection": True,
                "protection_type": "just_placed",
                "stop_price": None,
                "tp_price": None,
                "qty": position["qty"],
                "avg_entry": position["avg_entry_price"],
                "pnl_pct": position.get("unrealized_plpc", 0),
            }

        has_stop = False
        has_tp = False
        has_trailing = False
        protection_type = "none"
        stop_price = None
        tp_price = None
        stop_qty = 0.0
        trailing_qty = 0.0

        for o in orders:
            if o.get("side") != "sell":
                continue
            order_type = o.get("type", "")
            order_class = o.get("order_class", "simple")
            o_qty = float(o.get("qty", 0) or 0)

            if order_type in ("stop", "stop_limit"):
                has_stop = True
                stop_price = o.get("stop_price")
                stop_qty += o_qty
            elif order_type == "limit" and order_class in ("oco", "bracket"):
                has_tp = True
                tp_price = o.get("limit_price")
            elif order_type == "limit" and order_class in ("simple", "None", None, ""):
                if self._is_crypto_symbol(symbol):
                    has_tp = True
                    tp_price = o.get("limit_price")
            elif order_type == "trailing_stop":
                has_trailing = True
                trailing_qty += o_qty

        if has_stop and has_tp:
            protection_type = "bracket/oco"
        elif has_trailing:
            protection_type = "trailing_stop"
        elif has_stop:
            protection_type = "stop_only"
        elif has_tp:
            protection_type = "tp_only"

        entry = position["avg_entry_price"]
        current = position.get("current_price", entry)
        stop_distance_pct = 0.0
        if stop_price and entry > 0:
            # Measure staleness from CURRENT price, because stops are placed
            # relative to current price. Using entry would flag underwater
            # positions as "stale" even when the stop is correctly positioned
            # relative to the market. A stop 5% below current is never stale.
            reference_price = current if current > 0 else entry
            stop_distance_pct = round(abs(reference_price - stop_price) / reference_price * 100, 2)

        pos_qty = position["qty"]
        protected_qty = stop_qty + trailing_qty
        issues = []

        if (has_stop or has_trailing) and protected_qty > 0:
            if protected_qty < pos_qty * 0.95:
                issues.append(f"qty_gap:{pos_qty - protected_qty:.2f}")
            if has_stop and stop_distance_pct > self.STOP_LOSS_PCT * 2.0:
                issues.append(f"stale_stop:{stop_distance_pct:.1f}%")

        return {
            "symbol": symbol,
            "has_protection": has_stop or has_trailing,
            "protection_type": protection_type,
            "stop_price": stop_price,
            "tp_price": tp_price,
            "qty": pos_qty,
            "avg_entry": entry,
            "pnl_pct": position.get("unrealized_plpc", 0),
            "stop_qty": protected_qty,
            "stop_distance_pct": stop_distance_pct,
            "issues": issues,
        }

    def _add_oco_protection(self, symbol: str, position: Dict) -> bool:
        """Add OCO (stop-loss + take-profit) protection for equity positions."""
        is_crypto = self._is_crypto_symbol(symbol) or (
            position.get("asset_class", "") == "crypto"
        )

        if is_crypto:
            return self._add_crypto_protection(symbol, position)

        try:
            qty = position["qty_available"]
            total_qty = position["qty"]

            # Cancel and replace if qty_available is significantly less than total.
            # This happens when existing orders hold some shares (e.g., old bracket
            # legs from a previous ADD), leaving a qty_gap in protection.
            needs_full_replace = (
                (qty <= 0 and total_qty > 0) or
                (total_qty > 0 and qty < total_qty * 0.95)
            )

            if needs_full_replace:
                logger.info(f"  {symbol}: qty_available={qty:.1f} < total={total_qty:.1f} -- canceling all sells for unified OCO...")
                orders = self.get_open_orders()
                canceled = 0
                for o in orders:
                    if o["symbol"] == symbol and o["side"] == "sell":
                        try:
                            self.trading_client.cancel_order_by_id(o["id"])
                            canceled += 1
                        except Exception:
                            pass
                if canceled > 0:
                    logger.info(f"    Canceled {canceled} order(s) for {symbol}")
                time.sleep(3)  # Extra wait for Alpaca to release held qty
                qty = total_qty

            if qty <= 0:
                return False

            entry = position["avg_entry_price"]
            current = position.get("current_price", entry)
            # Use current price for stop if position is underwater,
            # to avoid stops that are too far from market and flagged as stale
            reference = current if current > 0 else entry
            sl_price = round(reference * (1 - self.STOP_LOSS_PCT / 100), 2)
            tp_price = round(entry * (1 + self.TAKE_PROFIT_PCT / 100), 2)

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OCO,
                limit_price=tp_price,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )

            order = self.trading_client.submit_order(request)
            logger.info(f"  OCO for {symbol}: SL ${sl_price} / TP ${tp_price} -> {order.id}")
            self._recently_protected.add(symbol)
            return True

        except Exception as e:
            logger.error(f"  OCO for {symbol} failed: {e}")
            self.log["errors"].append(f"OCO {symbol}: {e}")
            return False

    def _add_crypto_protection(self, symbol: str, position: Dict) -> bool:
        """Add stop-limit protection for crypto positions (no OCO support on Alpaca crypto)."""
        try:
            qty = position["qty_available"]
            total_qty = position["qty"]

            needs_full_replace = (
                (qty <= 0 and total_qty > 0) or
                (total_qty > 0 and qty < total_qty * 0.95)
            )

            if needs_full_replace:
                logger.info(f"  {symbol}: crypto qty_available={qty} < total={total_qty} -- canceling to replace...")
                orders = self.get_open_orders()
                sym_variants = self._crypto_symbol_variants(symbol)
                canceled = 0
                for o in orders:
                    if o["symbol"] in sym_variants and o["side"] == "sell":
                        try:
                            self.trading_client.cancel_order_by_id(o["id"])
                            canceled += 1
                        except Exception:
                            pass
                if canceled > 0:
                    logger.info(f"    Canceled {canceled} order(s) for {symbol}")
                time.sleep(3)
                qty = total_qty

            if qty <= 0:
                return False

            order_symbol = self._normalize_crypto_symbol(symbol)
            entry = position["avg_entry_price"]
            current = position.get("current_price", entry)

            # Calculate stop from CURRENT price (not entry) to avoid stale stop loops.
            # If position is underwater, entry-based stop would be too far from current
            # and would get flagged as "stale" on the next audit pass.
            reference_price = current if current > 0 else entry
            sl_price = round(reference_price * (1 - self.STOP_LOSS_PCT / 100), 2)
            sl_limit = round(sl_price * 0.995, 2)  # Slight buffer for slippage
            tp_price = round(entry * (1 + self.TAKE_PROFIT_PCT / 100), 2)

            # Safety: if stop is too close to current price, widen it
            if sl_price >= current * 0.99:
                sl_price = round(current * 0.93, 2)
                sl_limit = round(sl_price * 0.995, 2)

            placed_stop = False

            try:
                stop_request = StopLimitOrderRequest(
                    symbol=order_symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=sl_price,
                    limit_price=sl_limit,
                )
                stop_order = self.trading_client.submit_order(stop_request)
                logger.info(f"  CRYPTO STOP for {symbol}: trigger ${sl_price} / limit ${sl_limit} -> {stop_order.id}")
                placed_stop = True
            except ImportError:
                try:
                    stop_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        type="stop_limit",
                        stop_price=sl_price,
                        limit_price=sl_limit,
                    )
                    stop_order = self.trading_client.submit_order(stop_request)
                    logger.info(f"  CRYPTO STOP for {symbol}: trigger ${sl_price} / limit ${sl_limit} -> {stop_order.id}")
                    placed_stop = True
                except Exception as e2:
                    logger.error(f"  CRYPTO STOP fallback for {symbol} failed: {e2}")
                    self.log["errors"].append(f"Crypto stop {symbol}: {e2}")
            except Exception as e:
                logger.error(f"  CRYPTO STOP for {symbol} failed: {e}")
                self.log["errors"].append(f"Crypto stop {symbol}: {e}")

            if placed_stop:
                logger.info(f"  CRYPTO TP for {symbol}: skipped (Alpaca locks qty on stop) -- upside via graduation")
                self.log["actions"].append({
                    "symbol": symbol,
                    "action": "CRYPTO_PROTECTION",
                    "stop_price": sl_price,
                    "tp_price": None,
                    "note": "Separate orders (no OCO for crypto)",
                })
                self._recently_protected.add(symbol)
                for v in self._crypto_symbol_variants(symbol):
                    self._recently_protected.add(v)
            return placed_stop

        except Exception as e:
            logger.error(f"  Crypto protection for {symbol} failed completely: {e}")
            self.log["errors"].append(f"Crypto protection {symbol}: {e}")
            return False

    # --- TRAILING STOP GRADUATION ---------------------------------------------

    def _should_graduate(self, symbol: str, position: dict,
                         protection: dict) -> dict:
        """Determine if a position should graduate to trailing stop or tighten."""
        pnl_pct = position.get("unrealized_plpc", 0)
        current_type = protection.get("protection_type", "none")

        result = {
            "graduate": False,
            "reason": "",
            "trail_pct": self.TRAILING_STOP_TRAIL_PCT,
            "action": "none",
        }

        if current_type == "trailing_stop":
            if pnl_pct >= self.TRAILING_TIGHTEN_GAIN_PCT:
                current_trail = protection.get("trail_percent", self.TRAILING_STOP_TRAIL_PCT)
                if current_trail > self.TRAILING_TIGHTEN_TRAIL_PCT:
                    result["graduate"] = True
                    result["action"] = "tighten"
                    result["trail_pct"] = self.TRAILING_TIGHTEN_TRAIL_PCT
                    result["reason"] = (
                        f"Big winner ({pnl_pct:+.1f}%) -- tighten trail "
                        f"from {current_trail}% to {self.TRAILING_TIGHTEN_TRAIL_PCT}%"
                    )
                else:
                    result["reason"] = f"Already tight trailing ({pnl_pct:+.1f}%)"
            else:
                result["reason"] = f"Already trailing (gain: {pnl_pct:+.1f}%)"
            return result

        if current_type in ("bracket/oco", "stop_only", "tp_only"):
            if pnl_pct >= self.TRAILING_STOP_TRIGGER_GAIN_PCT:
                result["graduate"] = True
                result["action"] = "upgrade"
                result["trail_pct"] = self.TRAILING_STOP_TRAIL_PCT
                result["reason"] = (
                    f"Gain {pnl_pct:+.1f}% >= {self.TRAILING_STOP_TRIGGER_GAIN_PCT}% trigger -- "
                    f"graduating to {self.TRAILING_STOP_TRAIL_PCT}% trailing stop"
                )
            else:
                result["reason"] = (
                    f"Gain {pnl_pct:+.1f}% below {self.TRAILING_STOP_TRIGGER_GAIN_PCT}% trigger"
                )
            return result

        if current_type == "none":
            is_crypto = self._is_crypto_symbol(symbol)
            if is_crypto and pnl_pct >= self.TRAILING_STOP_TRIGGER_GAIN_PCT:
                result["graduate"] = True
                result["action"] = "upgrade"
                result["trail_pct"] = self.TRAILING_STOP_TRAIL_PCT
                result["reason"] = (
                    f"Crypto with no protection but gain {pnl_pct:+.1f}% >= "
                    f"{self.TRAILING_STOP_TRIGGER_GAIN_PCT}% -- placing trail-level stop"
                )
            else:
                result["reason"] = "No protection -- needs OCO first"
            return result

        return result

    def _upgrade_to_trailing_stop(self, symbol: str, position: dict,
                                   trail_pct: float, dry_run: bool = True) -> bool:
        """Replace existing protection with a trailing stop order."""
        qty = position["qty"]
        entry = position["avg_entry_price"]
        current = position["current_price"]
        pnl_pct = position.get("unrealized_plpc", 0)

        is_crypto = self._is_crypto_symbol(symbol) or position.get("asset_class", "") == "crypto"

        if dry_run:
            trail_stop = round(current * (1 - trail_pct / 100), 2)
            locked_gain_pct = round(((trail_stop / entry) - 1) * 100, 1)
            mode = "CRYPTO TRAIL" if is_crypto else "GRADUATE"
            print(f"    [DRY] {mode} {symbol}:")
            print(f"           Entry: ${entry:.2f} -> Current: ${current:.2f} ({pnl_pct:+.1f}%)")
            print(f"           Trail: {trail_pct}% -> Stop ~${trail_stop:.2f} (locks ~{locked_gain_pct:+.1f}%)")
            return True

        try:
            # Cancel existing sell orders first
            orders = self.get_open_orders()
            sym_variants = self._crypto_symbol_variants(symbol) if is_crypto else {symbol}
            for o in orders:
                if o["symbol"] in sym_variants and o["side"] == "sell":
                    try:
                        self.trading_client.cancel_order_by_id(o["id"])
                    except Exception:
                        pass
            time.sleep(2)

            if is_crypto:
                trail_stop = round(current * (1 - trail_pct / 100), 2)
                trail_limit = round(trail_stop * 0.995, 2)
                locked_gain_pct = round(((trail_stop / entry) - 1) * 100, 1)
                order_symbol = self._normalize_crypto_symbol(symbol)

                try:
                    request = StopLimitOrderRequest(
                        symbol=order_symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        stop_price=trail_stop,
                        limit_price=trail_limit,
                    )
                except ImportError:
                    request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        type="stop_limit",
                        stop_price=trail_stop,
                        limit_price=trail_limit,
                    )

                order = self.trading_client.submit_order(request)
                logger.info(
                    f"    CRYPTO TRAIL {symbol}: stop_limit at ${trail_stop} "
                    f"(locks ~{locked_gain_pct:+.1f}%) -> {order.id}"
                )

                self.log["actions"].append({
                    "symbol": symbol,
                    "action": "CRYPTO_TRAIL_UPGRADE",
                    "trail_pct": trail_pct,
                    "stop_price": trail_stop,
                    "locked_gain_pct": locked_gain_pct,
                    "order_id": str(order.id),
                    "pnl_at_graduation": round(pnl_pct, 2),
                    "note": "Simulated trailing via stop_limit (crypto)",
                })
                return True

            else:
                request = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=trail_pct,
                )

                order = self.trading_client.submit_order(request)
                trail_stop = round(current * (1 - trail_pct / 100), 2)
                locked_gain_pct = round(((trail_stop / entry) - 1) * 100, 1)

                logger.info(
                    f"    GRADUATED {symbol}: Trailing stop {trail_pct}% "
                    f"(stop ~${trail_stop:.2f}, locks ~{locked_gain_pct:+.1f}%) -> {order.id}"
                )

                self.log["actions"].append({
                    "symbol": symbol,
                    "action": "GRADUATE_TO_TRAILING",
                    "trail_pct": trail_pct,
                    "trail_stop_approx": trail_stop,
                    "locked_gain_pct": locked_gain_pct,
                    "order_id": str(order.id),
                    "pnl_at_graduation": round(pnl_pct, 2),
                })
                return True

        except Exception as e:
            logger.error(f"    Trailing stop for {symbol} failed: {e}")
            self.log["errors"].append(f"Trailing stop {symbol}: {e}")
            logger.info(f"    Restoring protection for {symbol}...")
            self._add_oco_protection(symbol, position)
            return False

    def graduate_profitable_positions(self, dry_run: bool = True) -> dict:
        """Graduate profitable positions from static stops to trailing stops."""
        print("\n  --- TRAILING STOP GRADUATION ---")

        positions = self.get_positions()
        orders = self.get_open_orders()
        orders_by_symbol = self._build_orders_by_symbol(orders)

        graduated = []
        tightened = []
        skipped = []

        for sym, pos in positions.items():
            if pos.get("is_option"):
                continue

            sym_orders = orders_by_symbol.get(sym, [])
            protection = self._classify_protection(sym, pos, sym_orders)

            for o in sym_orders:
                if o.get("trail_percent"):
                    protection["trail_percent"] = o["trail_percent"]

            decision = self._should_graduate(sym, pos, protection)

            if not decision["graduate"]:
                if pos.get("unrealized_plpc", 0) > 0:
                    skipped.append({
                        "symbol": sym,
                        "pnl_pct": round(pos.get("unrealized_plpc", 0), 2),
                        "reason": decision["reason"],
                    })
                continue

            action_label = "TIGHTEN" if decision["action"] == "tighten" else "GRADUATE"
            print(f"\n  {action_label} {sym}: {decision['reason']}")

            success = self._upgrade_to_trailing_stop(
                sym, pos, decision["trail_pct"], dry_run=dry_run
            )

            entry = {
                "symbol": sym,
                "action": decision["action"],
                "pnl_pct": round(pos.get("unrealized_plpc", 0), 2),
                "trail_pct": decision["trail_pct"],
                "status": "OK" if success else "FAILED",
            }

            if decision["action"] == "tighten":
                tightened.append(entry)
            else:
                graduated.append(entry)

            if not dry_run:
                time.sleep(1)

        summary = {
            "graduated": len(graduated),
            "tightened": len(tightened),
            "skipped": len(skipped),
            "details": {
                "graduated": graduated,
                "tightened": tightened,
                "skipped": skipped,
            }
        }

        print(f"\n  Graduation Summary:")
        print(f"     Graduated to trailing: {len(graduated)}")
        print(f"     Trail tightened: {len(tightened)}")
        print(f"     Skipped (not ready): {len(skipped)}")

        self.log["graduation"] = summary
        return summary

    def cancel_stale_orders(self, dry_run: bool = True) -> List[Dict]:
        """Cancel simple limit sell orders that provide no stop-loss protection."""
        orders = self.get_open_orders()
        positions = self.get_positions()
        canceled = []

        simple_sells = [
            o for o in orders
            if o["side"] == "sell"
            and o["type"] == "limit"
            and o.get("order_class", "simple") in ("simple", "None", None)
            and o["status"] in ("new", "accepted", "held")
        ]

        if not simple_sells:
            print("  No stale simple limit sells found")
            return canceled

        print(f"  Found {len(simple_sells)} simple limit sells (no stop-loss protection):")
        for o in simple_sells:
            sym = o["symbol"]
            pos = positions.get(sym)
            pos_info = f" (P&L: {pos['unrealized_plpc']:+.1f}%)" if pos else ""

            if dry_run:
                print(f"    [DRY] Would cancel: {sym} sell {o['qty']} @ ${o.get('limit_price', '?')}{pos_info}")
                canceled.append({"symbol": sym, "order_id": o["id"], "status": "DRY_RUN"})
            else:
                try:
                    self.trading_client.cancel_order_by_id(o["id"])
                    print(f"    Canceled: {sym} sell {o['qty']} @ ${o.get('limit_price', '?')}")
                    canceled.append({"symbol": sym, "order_id": o["id"], "status": "CANCELED"})
                    time.sleep(0.3)
                except Exception as e:
                    logger.error(f"    Cancel failed for {sym}: {e}")
                    canceled.append({"symbol": sym, "order_id": o["id"], "status": "FAILED"})

        return canceled

    def take_snapshot(self, label: str = "snapshot") -> str:
        """Save a timestamped portfolio snapshot."""
        today_dir = os.path.join(SNAPSHOTS_DIR, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(today_dir, exist_ok=True)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "account": self.get_account(),
            "positions": self.get_positions(),
            "open_orders": self.get_open_orders(),
        }

        filename = f"{label}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(today_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"  Snapshot saved: {filepath}")
        return filepath

    # --- MAIN WORKFLOWS -------------------------------------------------------

    def rebalance(self, dry_run: bool = True) -> Dict:
        """Full portfolio rebalance: score -> allocate -> execute -> protect -> graduate."""
        print_header("FULL PORTFOLIO REBALANCE")

        print("  Gathering portfolio state...")
        account = self.get_account()
        positions = self.get_positions()
        recommendations = self._load_recommendations()

        logger.info(f"  Equity: ${account['equity']:,.2f} | Cash: ${account['cash']:,.2f}")
        logger.info(f"  Positions: {len(positions)} | Daily P&L: {account['daily_pnl_pct']:+.1f}%")
        logger.info(f"  Recommendations loaded: {len(recommendations)} symbols")

        # Update drawdown tracker
        dd = update_drawdown(account["equity"])
        if dd.current_drawdown_pct > 0:
            logger.info(
                f"  Drawdown: {dd.current_drawdown_pct:.1f}% from peak ${dd.high_water_mark:,.0f} "
                f"(max ever: {dd.max_drawdown_pct:.1f}%)"
            )

        # Sector exposure report
        sector_exp = self._compute_sector_exposure(positions, account["equity"])
        if sector_exp:
            top_sectors = sorted(sector_exp.items(), key=lambda x: x[1], reverse=True)[:5]
            sector_str = " | ".join(f"{s}: {p:.1f}%" for s, p in top_sectors)
            logger.info(f"  Sector exposure: {sector_str}")

        if not dry_run:
            self.take_snapshot("pre_rebalance")

        print("\n  Cleaning stale orders...")
        self.cancel_stale_orders(dry_run=dry_run)

        print("\n  Computing target portfolio (volatility-adjusted + sector-aware)...")
        targets = self.compute_target_portfolio(account, positions, recommendations)
        self._print_target_summary(targets)

        print(f"\n  {'[DRY RUN] No real trades' if dry_run else 'EXECUTING LIVE TRADES'}...")
        executed = self.execute_plan(targets, account, positions, dry_run=dry_run)

        if not dry_run and executed:
            print("\n  Waiting for order fills to settle...")
            time.sleep(10)  # Bracket orders need extra time for all 3 legs

        print("\n  Auditing protection...")
        fresh_positions = positions if dry_run else self.get_positions()
        audit = self.audit_protection(fresh_positions, fix=True, dry_run=dry_run)

        if not dry_run and audit.get("unprotected", 0) > 0:
            logger.info(f"  Retrying protection for {audit['unprotected']} unprotected positions...")
            time.sleep(3)
            fresh_positions_2 = self.get_positions()
            audit = self.audit_protection(fresh_positions_2, fix=True, dry_run=False)

        print("\n  Checking for trailing stop graduations...")
        graduation = self.graduate_profitable_positions(dry_run=dry_run)

        if not dry_run:
            self.take_snapshot("post_rebalance")

        # Summary dashboard
        self._print_rebalance_dashboard(targets, executed, audit, graduation, account)

        self._save_log()

        return {
            "targets": targets,
            "executed": executed,
            "protection_audit": audit,
            "graduation": graduation,
            "account": account,
        }

    def new_trades(self, dry_run: bool = True) -> Dict:
        """Execute only new buy positions from AI recommendations."""
        print_header("EXECUTE NEW TRADES")

        account = self.get_account()
        positions = self.get_positions()
        recommendations = self._load_recommendations()

        held = set()
        for sym in positions.keys():
            held.add(sym)
            if self._is_crypto_symbol(sym):
                held.update(self._crypto_symbol_variants(sym))
        new_buys = {
            sym: rec for sym, rec in recommendations.items()
            if rec.get("action", "").upper() == "BUY"
            and sym not in held
            and rec.get("confidence", 0) >= self.MIN_CONFIDENCE_THRESHOLD
        }

        logger.info(f"  Equity: ${account['equity']:,.2f}")
        logger.info(f"  Current positions: {len(positions)} | Available slots: {self.MAX_POSITIONS - len(positions)}")
        logger.info(f"  New buy candidates: {len(new_buys)}")

        if not new_buys:
            print("  No new trade opportunities meet criteria")
            return {"executed": [], "protection_audit": {}}

        equity = account["equity"]
        targets = []
        available_slots = self.MAX_POSITIONS - len(positions)

        for sym, rec in sorted(new_buys.items(),
                                key=lambda x: x[1].get("confidence", 0),
                                reverse=True)[:available_slots]:
            score = self.score_opportunity(rec)
            atr_pct = rec.get("indicators", {}).get("atr_percent", 2.5)
            target_pct = self._size_for_score(score, atr_pct)
            targets.append({
                "symbol": sym,
                "action": "BUY",
                "score": score,
                "current_pct": 0.0,
                "target_pct": target_pct,
                "confidence": rec.get("confidence", 0),
                "is_existing": False,
                "atr_pct": atr_pct,
                "sector": rec.get("sector", get_sector(sym)),
            })

        executed = self.execute_plan(targets, account, positions, dry_run=dry_run)

        if not dry_run and executed:
            time.sleep(5)
        audit = self.audit_protection(fix=True, dry_run=dry_run)

        self._save_log()
        return {"executed": executed, "protection_audit": audit}

    def protect_only(self, dry_run: bool = True) -> Dict:
        """Fix unprotected positions and graduate winners. No new trades."""
        print_header("FIX POSITION PROTECTION")

        print("  Cleaning stale orders...")
        self.cancel_stale_orders(dry_run=dry_run)
        if not dry_run:
            time.sleep(2)

        print("\n  Auditing and fixing protection...")
        audit = self.audit_protection(fix=True, dry_run=dry_run)

        print("\n  Checking for trailing stop graduations...")
        graduation = self.graduate_profitable_positions(dry_run=dry_run)

        self._print_audit_summary(audit)
        self._save_log()
        return {"protection_audit": audit, "graduation": graduation}

    def audit(self) -> Dict:
        """Read-only protection audit."""
        print_header("PROTECTION AUDIT (READ-ONLY)")
        audit = self.audit_protection(fix=False, dry_run=True)
        self._print_audit_summary(audit)
        return {"protection_audit": audit}

    # --- HELPERS --------------------------------------------------------------

    def _load_recommendations(self) -> Dict[str, Dict]:
        """Load AI recommendations from the data directory."""
        data = load_json("recommendations.json")
        if not data:
            logger.warning("No recommendations.json found")
            return {}

        recs = {}
        for key in ("buy_signals", "sell_signals", "watch_signals"):
            for rec in data.get(key, []):
                sym = rec.get("symbol")
                if sym:
                    recs[sym] = rec
        return recs

    def _order_to_dict(self, order) -> Dict:
        """Convert an Alpaca order object to a plain dictionary."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
            "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "order_class": (
                order.order_class.value
                if order.order_class and hasattr(order.order_class, 'value')
                else str(order.order_class) if order.order_class else "simple"
            ),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "trail_percent": float(order.trail_percent) if order.trail_percent else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "legs": [self._order_to_dict(leg) for leg in order.legs] if order.legs else [],
        }

    def _print_target_summary(self, targets: List[Dict]):
        """Print the target portfolio summary table."""
        sells = [t for t in targets if t["action"] in ("SELL", "TRIM")]
        holds = [t for t in targets if t["action"] == "HOLD"]
        adds = [t for t in targets if t["action"] == "ADD"]
        buys = [t for t in targets if t["action"] == "BUY"]

        print(f"\n  {'Symbol':8} {'Action':6} {'Score':>5} {'Current':>8} {'Target':>8} {'P&L':>7} {'Sector':>12}")
        print(f"  {'-' * 62}")

        for t in sells:
            pnl = f"{t.get('pnl_pct', 0):+.1f}%" if t.get("is_existing") else ""
            sector = t.get("sector", "")[:11]
            print(f"  {t['symbol']:8} {'SELL/TRIM':10} {t['score']:>5.0f} "
                  f"{t['current_pct']:>7.1f}% {t['target_pct']:>7.1f}% {pnl:>7} {sector:>12}")

        for t in holds + adds:
            pnl = f"{t.get('pnl_pct', 0):+.1f}%" if t.get("is_existing") else ""
            icon = "HOLD" if t["action"] == "HOLD" else "ADD"
            sector = t.get("sector", "")[:11]
            print(f"  {t['symbol']:8} {icon:10} {t['score']:>5.0f} "
                  f"{t['current_pct']:>7.1f}% {t['target_pct']:>7.1f}% {pnl:>7} {sector:>12}")

        for t in buys:
            sector = t.get("sector", "")[:11]
            print(f"  {t['symbol']:8} {'BUY':10} {t['score']:>5.0f} "
                  f"{'0.0':>7}% {t['target_pct']:>7.1f}% {'':>7} {sector:>12}")

        total = sum(t["target_pct"] for t in targets if t["action"] != "SELL")
        cash = 100.0 - total
        print(f"\n  Total invested: {total:.1f}% | Cash reserve: {cash:.1f}%")
        print(f"  Actions: {len(sells)} sells/trims, {len(holds)} holds, {len(adds)} adds, {len(buys)} new buys")

    def _print_rebalance_dashboard(self, targets, executed, audit, graduation, account):
        """Print the post-rebalance summary dashboard."""
        _sells = [t for t in targets if t["action"] in ("SELL", "TRIM")]
        _holds = [t for t in targets if t["action"] == "HOLD"]
        _adds  = [t for t in targets if t["action"] == "ADD"]
        _buys  = [t for t in targets if t["action"] == "BUY"]
        _total_inv = sum(t["target_pct"] for t in targets if t["action"] != "SELL")

        _grad_up = graduation.get("graduated", 0) if graduation else 0
        _grad_tight = graduation.get("tightened", 0) if graduation else 0

        _a_prot   = audit.get("protected", 0)
        _a_deg    = audit.get("degraded", 0)
        _a_unprot = audit.get("unprotected", 0)
        _a_fixed  = audit.get("fixed", 0)
        _a_total  = audit.get("total_positions", 0)
        _a_opts   = audit.get("options_excluded", 0)

        _eq = account["equity"]
        _ca = account["cash"]
        _dp = account["daily_pnl_pct"]

        print("\n" + "=" * 60)
        print("  REBALANCE COMPLETE -- SUMMARY DASHBOARD")
        print("=" * 60)

        print(f"\n  PORTFOLIO")
        print(f"    Equity:     ${_eq:>12,.2f}")
        print(f"    Cash:       ${_ca:>12,.2f}")
        print(f"    Invested:   {_total_inv:>11.1f}%")
        print(f"    Daily P&L:  {_dp:>+11.1f}%")

        dd = load_drawdown_state()
        if dd.high_water_mark > 0:
            print(f"    Drawdown:   {dd.current_drawdown_pct:>11.1f}%  (max: {dd.max_drawdown_pct:.1f}%)")

        print(f"\n  TRADES")
        print(f"    Sold/Trimmed:  {len(_sells):>3}   "
              f"Held: {len(_holds):>3}   "
              f"Added: {len(_adds):>3}   "
              f"Bought: {len(_buys):>3}")
        if executed:
            print(f"    Orders placed: {len(executed)}")

        print(f"\n  PROTECTION ({_a_prot + _a_deg}/{_a_total} positions)")
        print(f"    Fully covered: {_a_prot:>3}")
        if _a_deg > 0:
            print(f"    Degraded:      {_a_deg:>3}  (qty gap or stale stop)")
        if _a_unprot > 0:
            print(f"    Unprotected:   {_a_unprot:>3}")
        if _a_fixed > 0:
            print(f"    Fixed this run: {_a_fixed:>2}")
        if _a_opts > 0:
            print(f"    Options (M06):  {_a_opts:>2}")

        print(f"\n  GRADUATION")
        print(f"    Upgraded to trailing: {_grad_up}")
        if _grad_tight > 0:
            print(f"    Trails tightened:     {_grad_tight}")

        # Health checks
        _health = []
        if _a_unprot > 0:
            _health.append(f"{_a_unprot} unprotected positions")
        if _a_deg > 0:
            _health.append(f"{_a_deg} degraded stops")
        if _dp < -self.DAILY_LOSS_LIMIT_PCT:
            _health.append("daily loss limit breached")
        _investable = 100 - self.CASH_RESERVE_PCT
        if _total_inv < _investable * 0.65:
            _health.append(f"under-invested ({_total_inv:.0f}% vs {_investable:.0f}% target)")
        if dd.current_drawdown_pct > MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT * 0.7:
            _health.append(f"drawdown warning ({dd.current_drawdown_pct:.1f}%)")

        if _health:
            _h_str = ", ".join(_health)
            print(f"\n  WARNING: {_h_str}")
        else:
            print(f"\n  HEALTH: All checks passed")
        print("=" * 60)

    def _print_audit_summary(self, audit: Dict):
        """Print protection audit results."""
        total = audit["total_positions"]
        prot = audit["protected"]
        deg = audit.get("degraded", 0)
        unprot = audit["unprotected"]
        fixed = audit["fixed"]
        options_ex = audit.get("options_excluded", 0)

        if total == 0:
            print("  No positions to audit")
            return

        pct = ((prot + deg) / total * 100) if total > 0 else 0
        icon = "OK" if unprot == 0 else "WARNING"

        print(f"\n  [{icon}] Protection: {prot}/{total} positions covered ({pct:.0f}%)")
        if deg > 0:
            print(f"     Degraded: {deg} positions (have stops but quality issues):")
            for d in audit.get("details", {}).get("degraded", []):
                _di = ", ".join(d.get("issues", []))
                _sp = d.get("stop_price", "?")
                _sd = d.get("stop_distance_pct", 0)
                _sq = d.get("stop_qty", 0)
                _pq = d.get("qty", 0)
                # Use higher precision for tiny crypto quantities
                if _pq < 1:
                    qty_fmt = f"{_sq:.6f}/{_pq:.6f}"
                else:
                    qty_fmt = f"{_sq:.1f}/{_pq:.1f}"
                print(f"        {d['symbol']:8s}  stop=${_sp:>8}  "
                      f"dist={_sd:.1f}%  qty={qty_fmt}  [{_di}]")
        if unprot > 0:
            _up_list = ", ".join(audit["details"]["unprotected"])
            print(f"     Unprotected: {_up_list}")
        if fixed > 0:
            _fixed_details = audit.get("details", {}).get("fixed", [])
            _statuses = {}
            for _fi in _fixed_details:
                _s = _fi.get("status", "?")
                _statuses[_s] = _statuses.get(_s, 0) + 1
            _status_str = ", ".join(f"{v} {k.lower()}" for k, v in _statuses.items())
            print(f"     Fixed: {fixed} positions ({_status_str})")
        if options_ex > 0:
            print(f"     Options excluded: {options_ex} (managed by Module 06)")

    def _save_log(self):
        """Save execution log to disk."""
        self.log["completed"] = datetime.now().isoformat()
        log_file = os.path.join(LOGS_DIR, f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        save_json("last_execution.json", self.log)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Execution Engine")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rebalance", action="store_true", help="Full portfolio rebalance")
    mode.add_argument("--new-trades", action="store_true", help="Only add new positions")
    mode.add_argument("--protect-only", action="store_true", help="Fix unprotected positions")
    mode.add_argument("--graduate", action="store_true", help="Graduate profitable positions to trailing stops")
    mode.add_argument("--audit", action="store_true", help="Read-only protection check")

    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    parser.add_argument("--force", action="store_true", help="Skip confirmations")

    args = parser.parse_args()
    engine = ExecutionEngine()

    if args.rebalance:
        engine.rebalance(dry_run=args.dry_run)
    elif args.new_trades:
        engine.new_trades(dry_run=args.dry_run)
    elif args.protect_only:
        engine.protect_only(dry_run=args.dry_run)
    elif args.graduate:
        engine.graduate_profitable_positions(dry_run=args.dry_run)
    elif args.audit:
        engine.audit()

    return 0


if __name__ == "__main__":
    sys.exit(main())