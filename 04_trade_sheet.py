#!/usr/bin/env python3
"""
Module 4: Human-Readable Trade Execution Sheet
================================================
Reads the output from Modules 0-2 (regime, analysis, portfolio) and generates
a clear, printable trade execution sheet that a human can follow step by step.

Outputs:
  - data/trade_sheet.txt       (plain text, printable)
  - data/trade_sheet.json      (structured, for reference)
  - Console output             (colored summary)

Run AFTER Module 1 (market analysis) and Module 2 (portfolio status).

Usage:
  python 04_trade_sheet.py                  # Generate sheet
  python 04_trade_sheet.py --detailed       # Include full reasoning
  python 04_trade_sheet.py --compact        # One-page summary only
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
    Designed to be printed and used as a manual checklist.
    """

    def __init__(self):
        self.cfg = Config()
        self.regime: Optional[RegimeContext] = load_regime_context()
        self.recommendations: Dict = load_json("recommendations.json") or {}
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
            r = self.regime.risk_appetite
            self.max_position_pct = round(3.0 + r * 5.0, 1)
            self.med_position_pct = round(2.0 + r * 3.5, 1)
            self.min_position_pct = round(1.0 + r * 2.0, 1)
        else:
            self.stop_loss_pct = 7.0
            self.take_profit_pct = 20.0
            self.max_positions = 20
            self.cash_reserve_pct = 3.0
            self.max_position_pct = 7.0
            self.med_position_pct = 4.5
            self.min_position_pct = 2.5

        # Load positions
        raw_positions = self.portfolio.get("positions", {})
        if isinstance(raw_positions, list):
            self.positions = {p.get("symbol", "?"): p for p in raw_positions if isinstance(p, dict)}
        elif isinstance(raw_positions, dict):
            self.positions = raw_positions
        else:
            self.positions = {}

    # -------------------------------------------------------------------------
    # TRADE PLAN COMPUTATION
    # -------------------------------------------------------------------------

    def _get_buy_signals(self) -> List[Dict]:
        """Get all actionable BUY signals sorted by confidence."""
        signals = self.recommendations.get("buy_signals", [])
        # Filter to actionable ones with price data
        actionable = []
        for sig in signals:
            conf = sig.get("confidence", 0)
            price = sig.get("current_price")
            symbol = sig.get("symbol", "")
            if conf >= 0.45 and price and price > 0 and symbol:
                actionable.append(sig)
        return sorted(actionable, key=lambda x: x.get("confidence", 0), reverse=True)

    def _get_sell_signals(self) -> List[Dict]:
        """Get all SELL signals for current holdings."""
        signals = self.recommendations.get("sell_signals", [])
        # Also check for positions that should be sold based on recommendations
        sell_list = []
        for sig in signals:
            symbol = sig.get("symbol", "")
            if symbol and symbol in self.positions:
                sig["position"] = self.positions[symbol]
                sell_list.append(sig)
        return sell_list

    def _compute_position_size(self, confidence: float, conviction_tier: str = "base") -> float:
        """Compute target position size as % of equity."""
        if confidence >= 0.72:
            return self.max_position_pct
        elif confidence >= 0.58:
            return self.med_position_pct
        else:
            return self.min_position_pct

    def _compute_shares(self, dollar_amount: float, price: float, is_crypto: bool = False) -> float:
        """Compute number of shares/units to buy."""
        if price <= 0:
            return 0
        if is_crypto:
            return round(dollar_amount / price, 6)
        else:
            return int(dollar_amount / price)

    def build_trade_plan(self) -> Dict:
        """Build the complete human-readable trade plan."""
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
            "buys": [],
            "sells": [],
            "holds": [],
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

        # Already-held symbols
        held_symbols = set(self.positions.keys())

        # ---- SELLS ----
        sell_signals = self._get_sell_signals()
        for sig in sell_signals:
            pos = sig.get("position", {})
            symbol = sig.get("symbol", "")
            qty = pos.get("qty", 0) or pos.get("quantity", 0)
            current_price = pos.get("current_price", 0) or sig.get("current_price", 0)
            pnl_pct = pos.get("unrealized_plpc", 0)
            market_value = pos.get("market_value", 0)

            plan["sells"].append({
                "symbol": symbol,
                "action": "SELL ALL",
                "shares": float(qty),
                "current_price": float(current_price),
                "market_value": float(market_value),
                "pnl_pct": float(pnl_pct),
                "order_type": "Market Sell",
                "confidence": sig.get("confidence", 0),
                "reasoning": sig.get("reasoning", ""),
                "catalyst": sig.get("catalyst", ""),
                "sector": sig.get("sector", ""),
            })

        # ---- BUYS ----
        buy_signals = self._get_buy_signals()
        cash_reserve = self.equity * (self.cash_reserve_pct / 100)
        available_capital = max(0, self.cash - cash_reserve)

        # Skip symbols we already own
        new_buys = [b for b in buy_signals if b.get("symbol", "") not in held_symbols]

        # Track how much capital we're allocating
        allocated = 0.0
        max_new_positions = self.max_positions - len(self.positions)

        for sig in new_buys:
            if len(plan["buys"]) >= max_new_positions:
                break

            symbol = sig.get("symbol", "")
            price = sig.get("current_price", 0)
            confidence = sig.get("confidence", 0)
            conviction = sig.get("conviction_tier", "base")
            is_crypto = sig.get("asset_type", "stock") == "crypto"

            # Position sizing
            target_pct = self._compute_position_size(confidence, conviction)
            target_dollars = self.equity * (target_pct / 100)

            # Don't exceed available capital
            if allocated + target_dollars > available_capital:
                target_dollars = max(0, available_capital - allocated)
                if target_dollars < 50:
                    continue

            shares = self._compute_shares(target_dollars, price, is_crypto)
            if shares <= 0:
                continue

            actual_cost = shares * price

            # Stop loss / take profit prices
            sl_pct = sig.get("stop_loss_pct") or self.stop_loss_pct
            tp_pct = sig.get("take_profit_pct") or self.take_profit_pct
            sl_price = round(price * (1 - sl_pct / 100), 2)
            tp_price = round(price * (1 + tp_pct / 100), 2)
            risk_per_share = round(price - sl_price, 2)
            reward_per_share = round(tp_price - price, 2)
            risk_reward = round(reward_per_share / risk_per_share, 1) if risk_per_share > 0 else 0
            max_loss = round(shares * risk_per_share, 2)

            plan["buys"].append({
                "priority": len(plan["buys"]) + 1,
                "symbol": symbol,
                "asset_type": "Crypto" if is_crypto else "Stock",
                "sector": sig.get("sector", ""),
                "action": "BUY",
                "order_type": "Bracket Order (Market + Stop-Loss + Take-Profit)",
                "shares": shares,
                "current_price": price,
                "total_cost": round(actual_cost, 2),
                "position_pct": round((actual_cost / self.equity) * 100, 1) if self.equity > 0 else 0,
                "stop_loss_price": sl_price,
                "stop_loss_pct": sl_pct,
                "take_profit_price": tp_price,
                "take_profit_pct": tp_pct,
                "risk_reward_ratio": f"{risk_reward}:1",
                "max_loss_dollars": max_loss,
                "confidence": confidence,
                "ai_agreement": sig.get("ai_agreement", "single"),
                "technical_bias": sig.get("technical_bias", "neutral"),
                "reasoning": sig.get("reasoning", ""),
                "catalyst": sig.get("catalyst", ""),
                "rs_vs_spy": sig.get("rs_vs_spy", 0) or (sig.get("indicators") or {}).get("rs_vs_spy", 0),
            })

            allocated += actual_cost

        # ---- HOLDS (current positions NOT being sold) ----
        sell_symbols = {s["symbol"] for s in plan["sells"]}
        for symbol, pos in self.positions.items():
            if symbol in sell_symbols:
                continue
            pnl_pct = pos.get("unrealized_plpc", 0)
            plan["holds"].append({
                "symbol": symbol,
                "shares": float(pos.get("qty", 0)),
                "avg_entry": float(pos.get("avg_entry_price", 0)),
                "current_price": float(pos.get("current_price", 0)),
                "market_value": float(pos.get("market_value", 0)),
                "pnl_pct": float(pnl_pct),
                "pnl_dollars": float(pos.get("unrealized_pl", 0)),
            })

        # ---- RISK SUMMARY ----
        total_new_investment = sum(b["total_cost"] for b in plan["buys"])
        total_max_loss = sum(b["max_loss_dollars"] for b in plan["buys"])
        remaining_cash = self.cash - total_new_investment

        plan["risk_summary"] = {
            "total_new_investment": round(total_new_investment, 2),
            "total_max_loss_new_trades": round(total_max_loss, 2),
            "max_loss_pct_of_equity": round((total_max_loss / self.equity) * 100, 2) if self.equity > 0 else 0,
            "remaining_cash_after_trades": round(remaining_cash, 2),
            "remaining_cash_pct": round((remaining_cash / self.equity) * 100, 1) if self.equity > 0 else 0,
            "new_positions_count": len(plan["buys"]),
            "sell_positions_count": len(plan["sells"]),
            "total_positions_after": len(self.positions) - len(plan["sells"]) + len(plan["buys"]),
            "cash_reserve_target": round(cash_reserve, 2),
        }

        return plan

    # -------------------------------------------------------------------------
    # TEXT OUTPUT
    # -------------------------------------------------------------------------

    def generate_text(self, plan: Dict, detailed: bool = False) -> str:
        """Generate the plain-text printable trade sheet."""
        lines = []
        W = 78  # page width

        def hr(char="="):
            lines.append(char * W)

        def center(text):
            lines.append(text.center(W))

        def blank():
            lines.append("")

        # ── HEADER ──
        hr()
        center("LUXVERUM CAPITAL — DAILY TRADE EXECUTION SHEET")
        center(plan["generated_display"])
        hr()
        blank()

        # ── MARKET REGIME ──
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

        # ── ACCOUNT SUMMARY ──
        hr("-")
        acct = plan["account"]
        lines.append(f"  ACCOUNT SUMMARY")
        lines.append(f"  Portfolio Equity:    ${acct['equity']:>12,.2f}")
        lines.append(f"  Available Cash:      ${acct['cash']:>12,.2f}")
        lines.append(f"  Buying Power:        ${acct['buying_power']:>12,.2f}")
        lines.append(f"  Current Positions:   {acct['current_positions']:>12}")
        blank()

        # ── SELLS FIRST ──
        sells = plan.get("sells", [])
        if sells:
            hr("=")
            center("STEP 1: SELL ORDERS (execute these FIRST)")
            hr("=")
            blank()
            lines.append(f"  {'#':>3}  {'Symbol':<10} {'Shares':>8} {'Price':>10} {'Value':>11} {'P&L':>8}  Order Type")
            lines.append(f"  {'---':>3}  {'------':<10} {'------':>8} {'-----':>10} {'-----':>11} {'---':>8}  ----------")

            for i, sell in enumerate(sells, 1):
                sym = sell["symbol"]
                shares = sell["shares"]
                price = sell["current_price"]
                value = sell["market_value"]
                pnl = sell["pnl_pct"]
                pnl_str = f"{pnl:+.1f}%"
                lines.append(
                    f"  {i:>3}  {sym:<10} {shares:>8.2f} ${price:>9.2f} ${value:>10,.2f} {pnl_str:>8}  {sell['order_type']}"
                )
                if detailed and sell.get("reasoning"):
                    lines.append(f"       Reason: {sell['reasoning'][:70]}")
            blank()

            lines.append("  INSTRUCTIONS: For each sell, place a MARKET SELL order for the full")
            lines.append("  quantity shown. These free up capital for new positions.")
            blank()
            lines.append(f"  [ ] Sell orders placed: ___ / {len(sells)}")
            blank()
        else:
            lines.append("  STEP 1: No sell orders today.")
            blank()

        # ── BUYS ──
        buys = plan.get("buys", [])
        if buys:
            hr("=")
            center("STEP 2: BUY ORDERS (with bracket protection)")
            hr("=")
            blank()

            for i, buy in enumerate(buys, 1):
                sym = buy["symbol"]
                atype = buy["asset_type"]
                shares = buy["shares"]
                price = buy["current_price"]
                cost = buy["total_cost"]
                sl = buy["stop_loss_price"]
                tp = buy["take_profit_price"]
                sl_pct = buy["stop_loss_pct"]
                tp_pct = buy["take_profit_pct"]
                rr = buy["risk_reward_ratio"]
                max_loss = buy["max_loss_dollars"]
                conf = buy["confidence"]
                agree = buy["ai_agreement"]
                tech = buy["technical_bias"]
                sector = buy["sector"]

                conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))

                lines.append(f"  ┌{'─' * (W - 4)}┐")
                lines.append(f"  │  #{i}  {sym:<12} ({atype} — {sector}){' ' * max(0, W - 6 - len(f'  #{i}  {sym:<12} ({atype} — {sector})') - 2)}│")
                lines.append(f"  ├{'─' * (W - 4)}┤")
                lines.append(f"  │  Confidence: [{conf_bar}] {conf:.0%}   AI Agreement: {agree:<8}  Tech: {tech:<10}│")
                lines.append(f"  │{' ' * (W - 4)}│")
                lines.append(f"  │  ORDER DETAILS:{' ' * (W - 4 - 17)}│")

                # Format shares display
                if atype == "Crypto":
                    share_str = f"{shares:.6f}"
                else:
                    share_str = f"{int(shares)}"

                line_buy   = f"    Action:       BUY {share_str} shares at market (~${price:,.2f})"
                line_cost  = f"    Total Cost:   ${cost:,.2f}  ({buy['position_pct']:.1f}% of portfolio)"
                line_sl    = f"    Stop-Loss:    ${sl:,.2f}  (−{sl_pct:.0f}% from entry)"
                line_tp    = f"    Take-Profit:  ${tp:,.2f}  (+{tp_pct:.0f}% from entry)"
                line_rr    = f"    Risk/Reward:  {rr}   |   Max Loss: ${max_loss:,.2f}"

                for line in [line_buy, line_cost, line_sl, line_tp, line_rr]:
                    padded = line + " " * max(0, W - 4 - len(line))
                    lines.append(f"  │{padded}│")

                if detailed:
                    lines.append(f"  │{' ' * (W - 4)}│")
                    # Wrap reasoning into lines
                    reasoning = buy.get("reasoning", "")
                    catalyst = buy.get("catalyst", "")
                    if reasoning:
                        label = "    Reasoning: "
                        self._wrap_text_in_box(lines, label, reasoning, W)
                    if catalyst:
                        label = "    Catalyst:  "
                        self._wrap_text_in_box(lines, label, catalyst, W)

                lines.append(f"  │{' ' * (W - 4)}│")
                chk_line = f"    [ ] Order placed   [ ] Bracket confirmed   [ ] Verified in account"
                padded_chk = chk_line + " " * max(0, W - 4 - len(chk_line))
                lines.append(f"  │{padded_chk}│")
                lines.append(f"  └{'─' * (W - 4)}┘")
                blank()

            # Buy totals
            risk = plan["risk_summary"]
            hr("-")
            lines.append(f"  BUY ORDERS SUMMARY:")
            lines.append(f"  Total New Investment:  ${risk['total_new_investment']:>12,.2f}")
            lines.append(f"  Total Max Loss (SLs):  ${risk['total_max_loss_new_trades']:>12,.2f}  ({risk['max_loss_pct_of_equity']:.1f}% of equity)")
            lines.append(f"  Cash After Trades:     ${risk['remaining_cash_after_trades']:>12,.2f}  ({risk['remaining_cash_pct']:.1f}% of equity)")
            lines.append(f"  Positions After:       {risk['total_positions_after']:>12}  (max: {self.max_positions})")
            blank()
            lines.append(f"  [ ] All buy orders placed: ___ / {len(buys)}")
            blank()
        else:
            lines.append("  STEP 2: No buy orders today.")
            blank()

        # ── CURRENT HOLDINGS ──
        holds = plan.get("holds", [])
        if holds:
            hr("=")
            center("CURRENT HOLDINGS (no action needed)")
            hr("-")
            lines.append(f"  {'Symbol':<10} {'Shares':>8} {'Entry':>10} {'Current':>10} {'Value':>11} {'P&L %':>8} {'P&L $':>10}")
            lines.append(f"  {'------':<10} {'------':>8} {'-----':>10} {'-------':>10} {'-----':>11} {'-----':>8} {'-----':>10}")

            total_value = 0
            total_pnl = 0
            for h in sorted(holds, key=lambda x: x["pnl_pct"], reverse=True):
                sym = h["symbol"]
                shares = h["shares"]
                entry = h["avg_entry"]
                curr = h["current_price"]
                val = h["market_value"]
                pnl_pct = h["pnl_pct"]
                pnl_dol = h["pnl_dollars"]
                total_value += val
                total_pnl += pnl_dol
                pnl_str = f"{pnl_pct:+.1f}%"
                pnl_d_str = f"${pnl_dol:+,.0f}"
                lines.append(
                    f"  {sym:<10} {shares:>8.2f} ${entry:>9.2f} ${curr:>9.2f} ${val:>10,.2f} {pnl_str:>8} {pnl_d_str:>10}"
                )

            lines.append(f"  {'─' * 70}")
            pnl_total_str = f"${total_pnl:+,.0f}"
            lines.append(
                f"  {'TOTAL':<10} {'':>8} {'':>10} {'':>10} ${total_value:>10,.2f} {'':>8} {pnl_total_str:>10}"
            )
            blank()

        # ── EXECUTION CHECKLIST ──
        hr("=")
        center("EXECUTION CHECKLIST")
        hr("-")
        lines.append("  Before trading:")
        lines.append("  [ ] Verify market is open (9:30 AM - 4:00 PM ET for stocks)")
        lines.append("  [ ] Check account has sufficient buying power")
        lines.append("  [ ] Review any overnight news that may change thesis")
        blank()
        lines.append("  Execution order:")
        if sells:
            lines.append(f"  [ ] 1. Execute all {len(sells)} SELL orders first")
            lines.append(f"  [ ] 2. Wait 1-2 minutes for sells to settle")
        if buys:
            step = 3 if sells else 1
            lines.append(f"  [ ] {step}. Execute BUY orders in priority order (highest confidence first)")
            lines.append(f"  [ ] {step+1}. For EACH buy: place as BRACKET order with SL + TP")
            lines.append(f"  [ ] {step+2}. Verify all bracket legs are active in open orders")
        blank()
        lines.append("  After trading:")
        lines.append("  [ ] Screenshot portfolio for records")
        lines.append("  [ ] Verify all stop-losses and take-profits are active")
        lines.append(f"  [ ] Confirm cash reserve >= ${plan['risk_summary'].get('cash_reserve_target', 0):,.2f}")
        blank()

        # ── RISK WARNINGS ──
        hr("=")
        center("RISK NOTES")
        hr("-")
        lines.append("  • All prices are approximate — use MARKET orders, not LIMIT, for entries")
        lines.append("  • Bracket orders = Market Buy + Stop-Loss + Take-Profit (OCO pair)")
        lines.append("  • If bracket order type is unavailable, place stop-loss IMMEDIATELY")
        lines.append(f"  • Never risk more than {plan['risk_summary'].get('max_loss_pct_of_equity', 0):.1f}% of equity on new trades")
        lines.append(f"  • Maintain at least {self.cash_reserve_pct:.0f}% cash reserve at all times")
        if self.regime and "bear" in self.regime.regime.lower():
            lines.append("  • BEAR REGIME: Smaller positions, tighter stops, prioritize capital preservation")
        elif self.regime and "bull" in self.regime.regime.lower():
            lines.append("  • BULL REGIME: Ride momentum, use wider stops, let winners run")
        blank()

        # ── FOOTER ──
        hr()
        center("Generated by Luxverum Capital AI Trading System")
        center(f"Module 04 — {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        center("This is NOT financial advice. Execute at your own discretion.")
        hr()

        return "\n".join(lines)

    def _wrap_text_in_box(self, lines: List[str], label: str, text: str, width: int):
        """Wrap long text inside the box border."""
        max_content = width - 6  # box borders + padding
        prefix_len = len(label)
        remaining = text

        first = True
        while remaining:
            if first:
                chunk_len = max_content - prefix_len
                chunk = remaining[:chunk_len]
                remaining = remaining[chunk_len:]
                full_line = label + chunk
                first = False
            else:
                chunk_len = max_content - prefix_len
                chunk = remaining[:chunk_len]
                remaining = remaining[chunk_len:]
                full_line = " " * prefix_len + chunk

            padded = full_line + " " * max(0, width - 4 - len(full_line))
            lines.append(f"  │{padded}│")

    # -------------------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------------------

    def generate(self, detailed: bool = False, compact: bool = False) -> Dict:
        """Generate the trade sheet and save outputs."""
        print_header("TRADE EXECUTION SHEET — Module 4")

        # Check we have data
        if not self.recommendations:
            print("  ✗ No recommendations found. Run Module 1 (market analysis) first.")
            return {}

        plan = self.build_trade_plan()

        # Generate text
        text_output = self.generate_text(plan, detailed=detailed)

        # Save text file
        txt_path = os.path.join(DATA_DIR, "trade_sheet.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_output)

        # Save JSON
        save_json("trade_sheet.json", plan)

        # Print to console
        print(text_output)

        # Quick summary
        print(f"\n  Files saved:")
        print(f"    {txt_path}")
        print(f"    {os.path.join(DATA_DIR, 'trade_sheet.json')}")

        return plan


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate human-readable trade execution sheet")
    parser.add_argument("--detailed", action="store_true", help="Include full AI reasoning for each trade")
    parser.add_argument("--compact", action="store_true", help="One-page compact summary only")
    args = parser.parse_args()

    generator = TradeSheetGenerator()
    plan = generator.generate(detailed=args.detailed, compact=args.compact)

    if not plan:
        return 1

    buys = len(plan.get("buys", []))
    sells = len(plan.get("sells", []))
    print(f"\n  Ready: {sells} sells, {buys} buys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
