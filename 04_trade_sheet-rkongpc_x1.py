#!/usr/bin/env python3
"""
Module 4: Human-Readable Trade Execution Sheet
================================================
Reads output from Modules 0-2 (regime, analysis, portfolio) and generates
a clear, printable trade execution sheet a human can follow step by step.

This mirrors the scoring logic from Module 3 (execution engine) so the
trade sheet matches exactly what the automated system would do.

Outputs:
  - data/trade_sheet.txt       (plain text, printable)
  - data/trade_sheet.json      (structured, for reference)
  - Console output             (colored summary)

Run AFTER Module 1 (market analysis) and Module 2 (portfolio status).

Usage:
  python 04_trade_sheet.py                  # Generate sheet
  python 04_trade_sheet.py --detailed       # Include full AI reasoning
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Shared config
try:
    from config import (
        Config, load_regime_context, RegimeContext,
        DATA_DIR, load_json, save_json, print_header,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        Config, load_regime_context, RegimeContext,
        DATA_DIR, load_json, save_json, print_header,
    )


# =============================================================================
# TRADE SHEET GENERATOR
# =============================================================================

class TradeSheetGenerator:
    """
    Generates a human-readable trade execution sheet from AI analysis output.
    Uses the same scoring logic as Module 3 (execution engine).
    """

    # ── Scoring weights (same as execution engine) ──
    W_AI_CONFIDENCE = 30
    W_TECHNICAL = 20
    W_MOMENTUM = 25
    W_PNL = 15
    W_TREND = 10

    SELL_SCORE_THRESHOLD = 30
    TRIM_THRESHOLD_PCT = 12.0

    def __init__(self):
        self.cfg = Config()
        self.regime: Optional[RegimeContext] = load_regime_context()
        self.raw_recommendations: Dict = load_json("recommendations.json") or {}
        self.portfolio: Dict = load_json("portfolio_status.json") or {}
        self.timestamp = datetime.now()

        # Derive account info
        account = self.portfolio.get("account", {})
        self.equity = float(account.get("equity", 0))
        self.cash = float(account.get("cash", 0))
        self.buying_power = float(account.get("buying_power", 0))

        # Derive regime parameters
        if self.regime:
            self.stop_loss_pct = self.regime.recommended_stop_loss_pct
            self.take_profit_pct = self.regime.recommended_take_profit_pct
            self.max_positions = self.regime.recommended_max_positions
            self.cash_reserve_pct = self.regime.recommended_cash_reserve_pct
            self.min_confidence = self.regime.recommended_confidence_threshold
            r = self.regime.risk_appetite
            self.max_position_pct = round(3.0 + r * 5.0, 1)
            self.med_position_pct = round(2.0 + r * 3.5, 1)
            self.min_position_pct = round(1.0 + r * 2.0, 1)
            self.high_conviction_threshold = 0.72
            self.med_conviction_threshold = 0.58
        else:
            self.stop_loss_pct = 7.0
            self.take_profit_pct = 20.0
            self.max_positions = 20
            self.cash_reserve_pct = 3.0
            self.min_confidence = 0.45
            self.max_position_pct = 7.0
            self.med_position_pct = 4.5
            self.min_position_pct = 2.5
            self.high_conviction_threshold = 0.72
            self.med_conviction_threshold = 0.58

        # Load positions
        raw_positions = self.portfolio.get("positions", {})
        if isinstance(raw_positions, list):
            self.positions = {p.get("symbol", "?"): p for p in raw_positions if isinstance(p, dict)}
        elif isinstance(raw_positions, dict):
            self.positions = raw_positions
        else:
            self.positions = {}

        # Build recommendations lookup (symbol -> rec)
        self.rec_lookup = self._build_rec_lookup()

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMENDATION LOOKUP
    # ─────────────────────────────────────────────────────────────────────────

    def _build_rec_lookup(self) -> Dict[str, Dict]:
        """Build a symbol -> recommendation dict from all signals."""
        lookup = {}
        for key in ("buy_signals", "sell_signals", "watch_signals", "recommendations"):
            signals = self.raw_recommendations.get(key, [])
            if isinstance(signals, list):
                for sig in signals:
                    sym = sig.get("symbol", "")
                    if sym and sym not in lookup:
                        lookup[sym] = sig
                    elif sym and sym in lookup:
                        # Keep the one with higher confidence
                        if sig.get("confidence", 0) > lookup[sym].get("confidence", 0):
                            lookup[sym] = sig
        return lookup

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING (mirrors Module 3 execution engine)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_existing_position(self, symbol: str, position: Dict) -> float:
        """Score an existing position (0-100). Same logic as execution engine."""
        rec = self.rec_lookup.get(symbol, {})
        score = 0.0

        # 1. AI Confidence (0-30 pts)
        confidence = rec.get("confidence", 0.0)
        action = rec.get("action", "").upper()
        if action == "SELL":
            score += 0
        elif action == "BUY":
            score += confidence * self.W_AI_CONFIDENCE
        else:
            score += confidence * self.W_AI_CONFIDENCE * 0.5  # WATCH = half

        # 2. Technical bias (0-20 pts)
        tech_bias = rec.get("technical_bias", "neutral").lower()
        tech_scores = {"bullish": 1.0, "mildly_bullish": 0.7, "neutral": 0.4,
                       "mildly_bearish": 0.2, "bearish": 0.0}
        score += tech_scores.get(tech_bias, 0.4) * self.W_TECHNICAL

        # 3. Momentum (0-25 pts) based on P&L
        pnl_pct = position.get("unrealized_plpc", 0)
        try:
            pnl_pct = float(pnl_pct)
        except (ValueError, TypeError):
            pnl_pct = 0
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

        # 4. P&L factor (0-15 pts)
        if pnl_pct > 0:
            score += min(self.W_PNL, pnl_pct * 0.5)
        else:
            score += max(0, self.W_PNL + pnl_pct * 0.5)

        # 5. Trend alignment (0-10 pts)
        if self.regime:
            regime_name = self.regime.regime.lower()
            if "bull" in regime_name:
                if pnl_pct > 0 and action != "SELL":
                    score += self.W_TREND
                else:
                    score += self.W_TREND * 0.3
            elif "bear" in regime_name:
                score += self.W_TREND * 0.2
            else:
                score += self.W_TREND * 0.5

        return round(min(100, max(0, score)), 1)

    def _score_new_opportunity(self, rec: Dict) -> float:
        """Score a new buy opportunity (0-100). Same logic as execution engine."""
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

        if self.regime and "bull" in self.regime.regime.lower():
            if confidence >= 0.70:
                score += 5

        return round(min(100, max(0, score)), 1)

    def _size_for_score(self, score: float) -> float:
        """Convert score to target position size %. Same as execution engine."""
        if score >= 80:
            return self.max_position_pct
        elif score >= 60:
            return self.med_position_pct
        else:
            return self.min_position_pct

    # ─────────────────────────────────────────────────────────────────────────
    # TRADE PLAN COMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def build_trade_plan(self) -> Dict:
        """Build the complete trade plan, mirroring execution engine logic."""
        plan = {
            "generated": self.timestamp.isoformat(),
            "generated_display": self.timestamp.strftime("%A, %B %d, %Y at %I:%M %p"),
            "account": {
                "equity": self.equity,
                "cash": self.cash,
                "buying_power": self.buying_power,
                "current_positions": len(self.positions),
            },
            "regime": None,
            "sells": [],
            "trims": [],
            "adds": [],
            "new_buys": [],
            "holds": [],
            "watch_list": [],
            "risk_summary": {},
        }

        # Regime info
        if self.regime:
            plan["regime"] = {
                "name": self.regime.regime.upper().replace("_", " "),
                "score": round(self.regime.regime_score, 3),
                "risk_appetite": round(self.regime.risk_appetite, 2),
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "cash_reserve_pct": self.cash_reserve_pct,
                "max_positions": self.max_positions,
                "position_range": f"{self.min_position_pct}% - {self.max_position_pct}%",
            }

        # ── SCORE EVERY EXISTING POSITION ──
        held_symbols = set()
        for symbol, pos in self.positions.items():
            held_symbols.add(symbol)
            score = self._score_existing_position(symbol, pos)
            rec = self.rec_lookup.get(symbol, {})

            qty = float(pos.get("qty", 0) or pos.get("quantity", 0))
            current_price = float(pos.get("current_price", 0))
            market_value = float(pos.get("market_value", 0))
            pnl_pct = float(pos.get("unrealized_plpc", 0))
            pnl_dollars = float(pos.get("unrealized_pl", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))
            current_pct = (abs(market_value) / self.equity * 100) if self.equity > 0 else 0

            base_info = {
                "symbol": symbol,
                "score": score,
                "shares": qty,
                "avg_entry": avg_entry,
                "current_price": current_price,
                "market_value": market_value,
                "position_pct": round(current_pct, 1),
                "pnl_pct": round(pnl_pct, 1),
                "pnl_dollars": round(pnl_dollars, 2),
                "sector": rec.get("sector", ""),
                "ai_action": rec.get("action", "N/A"),
                "ai_confidence": rec.get("confidence", 0),
                "technical_bias": rec.get("technical_bias", "N/A"),
                "reasoning": rec.get("reasoning", ""),
                "catalyst": rec.get("catalyst", ""),
            }

            if score < self.SELL_SCORE_THRESHOLD:
                # ── SELL ──
                base_info["action"] = "SELL ALL"
                base_info["order_type"] = "Market Sell"
                base_info["reason"] = f"Score {score:.0f}/100 (below {self.SELL_SCORE_THRESHOLD} threshold)"
                plan["sells"].append(base_info)

            elif current_pct > self.TRIM_THRESHOLD_PCT:
                # ── TRIM ──
                target_pct = self._size_for_score(score)
                target_value = self.equity * (target_pct / 100)
                trim_value = market_value - target_value
                trim_shares = int(trim_value / current_price) if current_price > 0 else 0
                base_info["action"] = "TRIM"
                base_info["order_type"] = "Market Sell (partial)"
                base_info["trim_shares"] = trim_shares
                base_info["trim_value"] = round(trim_value, 2)
                base_info["target_pct"] = round(target_pct, 1)
                base_info["reason"] = f"Position {current_pct:.1f}% exceeds {self.TRIM_THRESHOLD_PCT:.0f}% cap -> trim to {target_pct:.1f}%"
                if trim_shares > 0:
                    plan["trims"].append(base_info)
                else:
                    plan["holds"].append(base_info)

            elif score >= 70 and current_pct < self.med_position_pct:
                # ── ADD ──
                target_pct = self._size_for_score(score)
                add_value = self.equity * (target_pct / 100) - market_value
                if add_value > 50:  # Minimum $50
                    is_crypto = "/" in symbol
                    if is_crypto:
                        add_shares = round(add_value / current_price, 6) if current_price > 0 else 0
                    else:
                        add_shares = int(add_value / current_price) if current_price > 0 else 0

                    sl_pct = rec.get("stop_loss_pct") or self.stop_loss_pct
                    tp_pct = rec.get("take_profit_pct") or self.take_profit_pct
                    sl_price = round(current_price * (1 - sl_pct / 100), 2)
                    tp_price = round(current_price * (1 + tp_pct / 100), 2)

                    base_info["action"] = "ADD"
                    base_info["order_type"] = "Bracket Order"
                    base_info["add_shares"] = add_shares
                    base_info["add_value"] = round(add_shares * current_price, 2)
                    base_info["target_pct"] = round(target_pct, 1)
                    base_info["stop_loss_price"] = sl_price
                    base_info["stop_loss_pct"] = sl_pct
                    base_info["take_profit_price"] = tp_price
                    base_info["take_profit_pct"] = tp_pct
                    base_info["reason"] = f"Score {score:.0f}/100 -- strong hold, undersized at {current_pct:.1f}% -> add to {target_pct:.1f}%"
                    if add_shares > 0:
                        plan["adds"].append(base_info)
                    else:
                        plan["holds"].append(base_info)
                else:
                    plan["holds"].append(base_info)
            else:
                # ── HOLD ──
                plan["holds"].append(base_info)

        # ── SCORE NEW BUY OPPORTUNITIES (not already held) ──
        all_buy_recs = [r for r in self.rec_lookup.values()
                        if r.get("action", "").upper() == "BUY"
                        and r.get("symbol", "") not in held_symbols
                        and r.get("confidence", 0) >= self.min_confidence]

        for rec in all_buy_recs:
            score = self._score_new_opportunity(rec)
            if score < 40:
                continue

            symbol = rec["symbol"]
            price = rec.get("current_price", 0)
            if not price or price <= 0:
                continue

            target_pct = self._size_for_score(score)
            target_value = self.equity * (target_pct / 100)
            is_crypto = rec.get("asset_type", "stock") == "crypto" or "/" in symbol

            if is_crypto:
                shares = round(target_value / price, 6)
            else:
                shares = int(target_value / price)

            if shares <= 0:
                continue

            actual_cost = shares * price
            sl_pct = rec.get("stop_loss_pct") or self.stop_loss_pct
            tp_pct = rec.get("take_profit_pct") or self.take_profit_pct
            sl_price = round(price * (1 - sl_pct / 100), 2)
            tp_price = round(price * (1 + tp_pct / 100), 2)
            risk_per_share = round(price - sl_price, 2)
            reward_per_share = round(tp_price - price, 2)
            rr = round(reward_per_share / risk_per_share, 1) if risk_per_share > 0 else 0
            max_loss = round(shares * risk_per_share, 2)

            plan["new_buys"].append({
                "symbol": symbol,
                "score": score,
                "asset_type": "Crypto" if is_crypto else "Stock",
                "sector": rec.get("sector", ""),
                "action": "BUY NEW",
                "order_type": "Bracket Order (Market + Stop-Loss + Take-Profit)",
                "shares": shares,
                "current_price": price,
                "total_cost": round(actual_cost, 2),
                "position_pct": round((actual_cost / self.equity) * 100, 1) if self.equity > 0 else 0,
                "stop_loss_price": sl_price,
                "stop_loss_pct": sl_pct,
                "take_profit_price": tp_price,
                "take_profit_pct": tp_pct,
                "risk_reward_ratio": f"{rr}:1",
                "max_loss_dollars": max_loss,
                "confidence": rec.get("confidence", 0),
                "ai_agreement": rec.get("ai_agreement", "single"),
                "technical_bias": rec.get("technical_bias", "neutral"),
                "reasoning": rec.get("reasoning", ""),
                "catalyst": rec.get("catalyst", ""),
            })

        # Sort new buys by score
        plan["new_buys"].sort(key=lambda x: x["score"], reverse=True)

        # Cap new buys by available slots
        available_slots = max(0, self.max_positions - (len(self.positions) - len(plan["sells"])))
        plan["new_buys"] = plan["new_buys"][:available_slots] if available_slots > 0 else []

        # ── WATCH LIST (buy signals below threshold or no slots) ──
        watch_recs = [r for r in self.rec_lookup.values()
                      if r.get("action", "").upper() in ("BUY", "WATCH")
                      and r.get("symbol", "") not in held_symbols
                      and r.get("confidence", 0) >= 0.40]
        bought_symbols = {b["symbol"] for b in plan["new_buys"]}
        for rec in sorted(watch_recs, key=lambda x: x.get("confidence", 0), reverse=True)[:10]:
            sym = rec.get("symbol", "")
            if sym not in bought_symbols:
                plan["watch_list"].append({
                    "symbol": sym,
                    "confidence": rec.get("confidence", 0),
                    "sector": rec.get("sector", ""),
                    "reasoning": rec.get("reasoning", "")[:80],
                })

        # Sort holds by P&L
        plan["holds"].sort(key=lambda x: x.get("pnl_pct", 0), reverse=True)

        # ── RISK SUMMARY ──
        total_sell_value = sum(s["market_value"] for s in plan["sells"])
        total_trim_value = sum(t.get("trim_value", 0) for t in plan["trims"])
        total_add_cost = sum(a.get("add_value", 0) for a in plan["adds"])
        total_new_cost = sum(b["total_cost"] for b in plan["new_buys"])
        total_new_max_loss = sum(b["max_loss_dollars"] for b in plan["new_buys"])
        capital_freed = total_sell_value + total_trim_value
        capital_needed = total_add_cost + total_new_cost
        net_capital = capital_freed - capital_needed
        remaining_cash = self.cash + net_capital

        plan["risk_summary"] = {
            "capital_freed_sells": round(total_sell_value, 2),
            "capital_freed_trims": round(total_trim_value, 2),
            "capital_needed_adds": round(total_add_cost, 2),
            "capital_needed_new": round(total_new_cost, 2),
            "net_capital_change": round(net_capital, 2),
            "total_max_loss_new": round(total_new_max_loss, 2),
            "max_loss_pct_of_equity": round((total_new_max_loss / self.equity) * 100, 2) if self.equity > 0 else 0,
            "remaining_cash": round(remaining_cash, 2),
            "remaining_cash_pct": round((remaining_cash / self.equity) * 100, 1) if self.equity > 0 else 0,
            "cash_reserve_target": round(self.equity * self.cash_reserve_pct / 100, 2),
            "positions_before": len(self.positions),
            "positions_after": len(self.positions) - len(plan["sells"]) + len(plan["new_buys"]),
        }

        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT OUTPUT
    # ─────────────────────────────────────────────────────────────────────────

    def generate_text(self, plan: Dict, detailed: bool = False) -> str:
        """Generate the plain-text printable trade sheet."""
        lines = []
        W = 78

        def hr(char="="):
            lines.append(char * W)

        def center(text):
            lines.append(text.center(W))

        def blank():
            lines.append("")

        # ═══════════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════════
        hr()
        center("LUXVERUM CAPITAL -- DAILY TRADE EXECUTION SHEET")
        center(plan["generated_display"])
        hr()
        blank()

        # REGIME
        regime = plan.get("regime")
        if regime:
            lines.append(f"  MARKET REGIME:  {regime['name']}  (score: {regime['score']:+.3f})")
            lines.append(f"  Risk Appetite:  {regime['risk_appetite']:.0%}")
            lines.append(f"  Position Range: {regime['position_range']}  |  Max Positions: {regime['max_positions']}")
            lines.append(f"  Stop-Loss:      {regime['stop_loss_pct']}%  |  Take-Profit: {regime['take_profit_pct']}%")
            lines.append(f"  Cash Reserve:   {regime['cash_reserve_pct']}%")
        else:
            lines.append("  MARKET REGIME:  Not available (using defaults)")
        blank()

        # ACCOUNT
        hr("-")
        acct = plan["account"]
        lines.append(f"  ACCOUNT SUMMARY")
        lines.append(f"  Portfolio Equity:    ${acct['equity']:>12,.2f}")
        lines.append(f"  Available Cash:      ${acct['cash']:>12,.2f}")
        lines.append(f"  Buying Power:        ${acct['buying_power']:>12,.2f}")
        lines.append(f"  Current Positions:   {acct['current_positions']:>12}")
        blank()

        step_num = 0

        # ═══════════════════════════════════════════════════════════════════════
        # STEP: SELLS
        # ═══════════════════════════════════════════════════════════════════════
        sells = plan.get("sells", [])
        step_num += 1
        if sells:
            hr("=")
            center(f"STEP {step_num}: SELL ORDERS -- execute FIRST to free capital")
            hr("=")
            blank()

            lines.append(f"  {'#':>3}  {'Symbol':<10} {'Shares':>8} {'Price':>10} {'Value':>11} {'P&L':>8} {'Score':>6}")
            lines.append(f"  {'---':>3}  {'------':<10} {'------':>8} {'-----':>10} {'-----':>11} {'---':>8} {'-----':>6}")

            for i, sell in enumerate(sells, 1):
                lines.append(
                    f"  {i:>3}  {sell['symbol']:<10} {sell['shares']:>8.2f} "
                    f"${sell['current_price']:>9.2f} ${sell['market_value']:>10,.2f} "
                    f"{sell['pnl_pct']:>+7.1f}% {sell['score']:>5.0f}"
                )
                lines.append(f"       Reason: {sell['reason']}")
                if detailed and sell.get("reasoning"):
                    lines.append(f"       AI says: {sell['reasoning'][:70]}")
                blank()

            lines.append("  ORDER TYPE: Market Sell for full quantity.")
            lines.append(f"  [ ] Sell orders placed: ___ / {len(sells)}")
            blank()
        else:
            lines.append(f"  STEP {step_num}: SELL -- No positions score below {self.SELL_SCORE_THRESHOLD}/100. All holdings OK.")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # STEP: TRIMS
        # ═══════════════════════════════════════════════════════════════════════
        trims = plan.get("trims", [])
        step_num += 1
        if trims:
            hr("=")
            center(f"STEP {step_num}: TRIM OVERSIZED POSITIONS")
            hr("=")
            blank()

            for i, trim in enumerate(trims, 1):
                lines.append(f"  #{i}  {trim['symbol']}")
                lines.append(f"      Current: {trim['position_pct']:.1f}% of portfolio (${trim['market_value']:,.2f})")
                lines.append(f"      Target:  {trim['target_pct']:.1f}% of portfolio")
                lines.append(f"      Action:  SELL {trim['trim_shares']} shares (~${trim['trim_value']:,.2f})")
                lines.append(f"      Score:   {trim['score']:.0f}/100  |  P&L: {trim['pnl_pct']:+.1f}%")
                blank()

            lines.append(f"  [ ] Trim orders placed: ___ / {len(trims)}")
            blank()
        else:
            lines.append(f"  STEP {step_num}: TRIM -- No positions exceed {self.TRIM_THRESHOLD_PCT:.0f}% allocation cap.")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # STEP: ADDS (increase existing positions)
        # ═══════════════════════════════════════════════════════════════════════
        adds = plan.get("adds", [])
        step_num += 1
        if adds:
            hr("=")
            center(f"STEP {step_num}: ADD TO EXISTING POSITIONS (high-scoring, undersized)")
            hr("=")
            blank()

            for i, add in enumerate(adds, 1):
                sym = add["symbol"]
                conf_bar = "#" * int(add.get("ai_confidence", 0) * 10) + "." * (10 - int(add.get("ai_confidence", 0) * 10))

                lines.append(f"  +----------------------------------------------------------------------+")
                lines.append(f"  |  #{i}  {sym:<12} Score: {add['score']:.0f}/100   AI: [{conf_bar}] {add.get('ai_confidence',0):.0%}")
                lines.append(f"  +----------------------------------------------------------------------+")
                lines.append(f"  |  Currently hold: {add['shares']:.2f} shares @ ${add['avg_entry']:.2f} (P&L {add['pnl_pct']:+.1f}%)")
                lines.append(f"  |  Current size:   {add['position_pct']:.1f}% -> Target: {add['target_pct']:.1f}%")
                lines.append(f"  |")
                lines.append(f"  |  ACTION:  BUY {add['add_shares']} more shares at market (~${add['current_price']:,.2f})")
                lines.append(f"  |  Cost:    ${add['add_value']:,.2f}")
                lines.append(f"  |  SL:      ${add['stop_loss_price']:,.2f} (-{add['stop_loss_pct']:.0f}%)  |  TP: ${add['take_profit_price']:,.2f} (+{add['take_profit_pct']:.0f}%)")
                if detailed and add.get("reasoning"):
                    lines.append(f"  |  Reason:  {add['reasoning'][:65]}")
                lines.append(f"  |")
                lines.append(f"  |  [ ] Order placed   [ ] Bracket confirmed")
                lines.append(f"  +----------------------------------------------------------------------+")
                blank()

            lines.append(f"  [ ] Add orders placed: ___ / {len(adds)}")
            blank()
        else:
            lines.append(f"  STEP {step_num}: ADD -- No existing positions qualify for increase right now.")
            lines.append(f"           (need score >= 70 AND position < {self.med_position_pct:.1f}% allocation)")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # STEP: NEW BUYS
        # ═══════════════════════════════════════════════════════════════════════
        new_buys = plan.get("new_buys", [])
        step_num += 1
        if new_buys:
            hr("=")
            center(f"STEP {step_num}: NEW POSITION BUYS (with bracket protection)")
            hr("=")
            blank()

            for i, buy in enumerate(new_buys, 1):
                sym = buy["symbol"]
                atype = buy["asset_type"]
                conf_bar = "#" * int(buy["confidence"] * 10) + "." * (10 - int(buy["confidence"] * 10))

                if atype == "Crypto":
                    share_str = f"{buy['shares']:.6f}"
                else:
                    share_str = f"{int(buy['shares'])}"

                lines.append(f"  +----------------------------------------------------------------------+")
                lines.append(f"  |  #{i}  {sym:<12} ({atype} -- {buy['sector'][:25]})")
                lines.append(f"  |  Score: {buy['score']:.0f}/100   Confidence: [{conf_bar}] {buy['confidence']:.0%}")
                lines.append(f"  |  AI Agreement: {buy['ai_agreement']:<8}  |  Technical: {buy['technical_bias']}")
                lines.append(f"  |")
                lines.append(f"  |  ACTION:       BUY {share_str} shares at market (~${buy['current_price']:,.2f})")
                lines.append(f"  |  Total Cost:   ${buy['total_cost']:,.2f}  ({buy['position_pct']:.1f}% of portfolio)")
                lines.append(f"  |  Stop-Loss:    ${buy['stop_loss_price']:,.2f}  (-{buy['stop_loss_pct']:.0f}% from entry)")
                lines.append(f"  |  Take-Profit:  ${buy['take_profit_price']:,.2f}  (+{buy['take_profit_pct']:.0f}% from entry)")
                lines.append(f"  |  Risk/Reward:  {buy['risk_reward_ratio']}   |   Max Loss: ${buy['max_loss_dollars']:,.2f}")
                if detailed and buy.get("reasoning"):
                    lines.append(f"  |")
                    lines.append(f"  |  Reasoning: {buy['reasoning'][:65]}")
                if detailed and buy.get("catalyst"):
                    lines.append(f"  |  Catalyst:  {buy['catalyst'][:65]}")
                lines.append(f"  |")
                lines.append(f"  |  [ ] Order placed   [ ] Bracket confirmed   [ ] Verified")
                lines.append(f"  +----------------------------------------------------------------------+")
                blank()

            lines.append(f"  [ ] New buy orders placed: ___ / {len(new_buys)}")
            blank()
        else:
            avail = max(0, self.max_positions - (len(self.positions) - len(sells)))
            if avail <= 0:
                lines.append(f"  STEP {step_num}: NEW BUYS -- No open slots.")
                lines.append(f"           Currently {len(self.positions)} positions (max {self.max_positions}).")
                lines.append(f"           Sell weak positions first to make room for new opportunities.")
            else:
                lines.append(f"  STEP {step_num}: NEW BUYS -- {avail} slots open but no new opportunities meet")
                lines.append(f"           the minimum confidence threshold ({self.min_confidence:.0%}) right now.")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # CAPITAL SUMMARY
        # ═══════════════════════════════════════════════════════════════════════
        risk = plan["risk_summary"]
        hr("=")
        center("CAPITAL & RISK SUMMARY")
        hr("-")
        lines.append(f"  Capital freed from sells:     ${risk['capital_freed_sells']:>11,.2f}")
        lines.append(f"  Capital freed from trims:     ${risk['capital_freed_trims']:>11,.2f}")
        lines.append(f"  Capital needed for adds:      ${risk['capital_needed_adds']:>11,.2f}")
        lines.append(f"  Capital needed for new buys:  ${risk['capital_needed_new']:>11,.2f}")
        lines.append(f"  Net capital change:           ${risk['net_capital_change']:>+11,.2f}")
        blank()
        if risk['total_max_loss_new'] > 0:
            lines.append(f"  Max loss on new trades (SLs): ${risk['total_max_loss_new']:>11,.2f}  ({risk['max_loss_pct_of_equity']:.1f}% of equity)")
        lines.append(f"  Cash after all trades:        ${risk['remaining_cash']:>11,.2f}  ({risk['remaining_cash_pct']:.1f}% of equity)")
        lines.append(f"  Cash reserve target:          ${risk['cash_reserve_target']:>11,.2f}  ({self.cash_reserve_pct:.0f}%)")
        lines.append(f"  Positions: {risk['positions_before']} -> {risk['positions_after']}  (max {self.max_positions})")
        blank()

        # ═══════════════════════════════════════════════════════════════════════
        # CURRENT HOLDINGS TABLE (with scores!)
        # ═══════════════════════════════════════════════════════════════════════
        holds = plan.get("holds", [])
        if holds:
            hr("=")
            center("CURRENT HOLDINGS -- HOLD (with scores)")
            hr("-")
            lines.append(f"  {'Symbol':<10} {'Score':>5} {'Shares':>8} {'Entry':>10} {'Now':>10} {'Value':>11} {'Alloc':>6} {'P&L':>8}")
            lines.append(f"  {'------':<10} {'-----':>5} {'------':>8} {'-----':>10} {'---':>10} {'-----':>11} {'-----':>6} {'---':>8}")

            total_value = 0
            total_pnl = 0
            for h in holds:
                total_value += h["market_value"]
                total_pnl += h.get("pnl_dollars", 0)
                lines.append(
                    f"  {h['symbol']:<10} {h['score']:>5.0f} {h['shares']:>8.2f} "
                    f"${h['avg_entry']:>9.2f} ${h['current_price']:>9.2f} "
                    f"${h['market_value']:>10,.2f} {h['position_pct']:>5.1f}% {h['pnl_pct']:>+7.1f}%"
                )

            lines.append(f"  {'':->78}")
            lines.append(
                f"  {'TOTAL':<10} {'':>5} {'':>8} {'':>10} {'':>10} "
                f"${total_value:>10,.2f} {'':>6} ${total_pnl:>+8,.0f}"
            )
            blank()
            lines.append(f"  Score guide: 70+ strong | 50-69 ok | 30-49 weak | <30 sell candidate")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # WATCH LIST
        # ═══════════════════════════════════════════════════════════════════════
        watch = plan.get("watch_list", [])
        if watch:
            hr("-")
            lines.append(f"  WATCH LIST (not trading today, monitor for entry):")
            for w in watch[:8]:
                lines.append(f"    {w['symbol']:<10} Conf: {w['confidence']:.0%}  {w['sector'][:20]:<20}  {w.get('reasoning', '')[:40]}")
            blank()

        # ═══════════════════════════════════════════════════════════════════════
        # EXECUTION CHECKLIST
        # ═══════════════════════════════════════════════════════════════════════
        hr("=")
        center("EXECUTION CHECKLIST")
        hr("-")
        lines.append("  Before trading:")
        lines.append("  [ ] Verify market is open (9:30 AM - 4:00 PM ET for stocks)")
        lines.append("  [ ] Check account has sufficient buying power")
        lines.append("  [ ] Review any overnight news that may change thesis")
        blank()

        exec_step = 1
        has_trades = sells or trims or adds or new_buys
        lines.append("  Execution order:")
        if sells:
            lines.append(f"  [ ] {exec_step}. Execute all {len(sells)} SELL orders first")
            exec_step += 1
            lines.append(f"  [ ] {exec_step}. Wait 1-2 minutes for sells to settle")
            exec_step += 1
        if trims:
            lines.append(f"  [ ] {exec_step}. Execute {len(trims)} TRIM orders")
            exec_step += 1
        if adds:
            lines.append(f"  [ ] {exec_step}. Execute {len(adds)} ADD orders (bracket with SL + TP)")
            exec_step += 1
        if new_buys:
            lines.append(f"  [ ] {exec_step}. Execute {len(new_buys)} NEW BUY orders in priority order")
            exec_step += 1
            lines.append(f"  [ ] {exec_step}. For EACH buy: place as BRACKET order with SL + TP")
            exec_step += 1
        if not has_trades:
            lines.append("       No trades today. Portfolio is fully positioned.")
        blank()

        lines.append("  After trading:")
        lines.append("  [ ] Screenshot portfolio for records")
        lines.append("  [ ] Verify all stop-losses and take-profits are active")
        lines.append(f"  [ ] Confirm cash reserve >= ${risk.get('cash_reserve_target', 0):,.2f}")
        blank()

        # RISK NOTES
        hr("=")
        center("RISK NOTES")
        hr("-")
        lines.append("  * All prices are approximate -- use MARKET orders for entries")
        lines.append("  * Bracket orders = Market Buy + Stop-Loss + Take-Profit (OCO pair)")
        lines.append("  * If bracket order unavailable, place stop-loss IMMEDIATELY after fill")
        if risk.get("max_loss_pct_of_equity", 0) > 0:
            lines.append(f"  * Max risk on new trades: {risk['max_loss_pct_of_equity']:.1f}% of equity")
        lines.append(f"  * Maintain at least {self.cash_reserve_pct:.0f}% cash reserve at all times")
        if self.regime:
            rn = self.regime.regime.lower()
            if "bear" in rn:
                lines.append("  * BEAR REGIME: Smaller positions, tighter stops, preserve capital")
            elif "bull" in rn:
                lines.append("  * BULL REGIME: Ride momentum, use wider stops, let winners run")
        blank()

        # FOOTER
        hr()
        center("Generated by Luxverum Capital AI Trading System")
        center(f"Module 04 -- {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        center("This is NOT financial advice. Execute at your own discretion.")
        hr()

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self, detailed: bool = False) -> Dict:
        """Generate the trade sheet and save outputs."""
        print_header("TRADE EXECUTION SHEET -- Module 4")

        if not self.raw_recommendations:
            print("  ERROR: No recommendations found. Run Module 1 (market analysis) first.")
            return {}
        if not self.positions and not self.raw_recommendations.get("buy_signals"):
            print("  WARNING: No positions and no buy signals found.")

        plan = self.build_trade_plan()
        text_output = self.generate_text(plan, detailed=detailed)

        # Save files
        txt_path = os.path.join(DATA_DIR, "trade_sheet.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_output)

        save_json("trade_sheet.json", plan)

        # Print to console
        print(text_output)

        print(f"\n  Files saved:")
        print(f"    {txt_path}")
        print(f"    {os.path.join(DATA_DIR, 'trade_sheet.json')}")

        # Quick tally
        ns = len(plan.get("sells", []))
        nt = len(plan.get("trims", []))
        na = len(plan.get("adds", []))
        nb = len(plan.get("new_buys", []))
        nh = len(plan.get("holds", []))
        nw = len(plan.get("watch_list", []))
        print(f"\n  Summary: {ns} sells | {nt} trims | {na} adds | {nb} new buys | {nh} holds | {nw} watching")

        return plan


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate human-readable trade execution sheet")
    parser.add_argument("--detailed", action="store_true", help="Include full AI reasoning for each trade")
    args = parser.parse_args()

    generator = TradeSheetGenerator()
    plan = generator.generate(detailed=args.detailed)
    return 0 if plan else 1


if __name__ == "__main__":
    sys.exit(main())