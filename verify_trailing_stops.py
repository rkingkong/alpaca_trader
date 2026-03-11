#!/usr/bin/env python3
"""
Trailing Stop Verification Tool
=================================
Queries Alpaca to verify all trailing stop orders and protection status.
Shows the complete protection map for every position.

Usage:
  python verify_trailing_stops.py              # Full verification
  python verify_trailing_stops.py --orders     # Show all open orders detail
  python verify_trailing_stops.py --history    # Show today's filled/canceled orders
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import pytz

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_config():
    """Load config from config.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TrailingStopVerifier:
    def __init__(self):
        config = load_config()
        self.trading_client = TradingClient(
            config.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY")),
            config.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY")),
            paper=config.get("PAPER_TRADING", True),
        )
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def get_positions(self):
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100,
            }
        return result

    def get_open_orders(self):
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        orders = self.trading_client.get_orders(filter=request)
        return orders

    def get_todays_closed_orders(self):
        """Get all orders that filled or were canceled today."""
        now = datetime.now(pytz.UTC)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            nested=True,
            after=start_of_day - timedelta(days=1),
        )
        orders = self.trading_client.get_orders(filter=request)
        return orders

    def _order_summary(self, order) -> dict:
        """Extract key fields from an order object."""
        return {
            "id": str(order.id)[:12] + "...",
            "full_id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
            "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
            "qty": float(order.qty) if order.qty else 0,
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "order_class": (
                order.order_class.value
                if order.order_class and hasattr(order.order_class, 'value')
                else str(order.order_class) if order.order_class else "simple"
            ),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "trail_percent": float(order.trail_percent) if order.trail_percent else None,
            "trail_price": float(order.trail_price) if order.trail_price else None,
            "hwm": float(order.hwm) if hasattr(order, 'hwm') and order.hwm else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": order.created_at.strftime("%Y-%m-%d %H:%M") if order.created_at else None,
            "filled_at": order.filled_at.strftime("%Y-%m-%d %H:%M") if order.filled_at else None,
            "canceled_at": order.canceled_at.strftime("%Y-%m-%d %H:%M") if order.canceled_at else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def verify_all(self):
        """Full verification of all positions and their protection orders."""
        print("\n" + "=" * 70)
        print("  TRAILING STOP & PROTECTION VERIFICATION")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70)

        positions = self.get_positions()
        open_orders = self.get_open_orders()

        # Build order map by symbol
        orders_by_symbol = defaultdict(list)
        for o in open_orders:
            summary = self._order_summary(o)
            orders_by_symbol[summary["symbol"]].append(summary)
            # Also include legs
            if o.legs:
                for leg in o.legs:
                    leg_summary = self._order_summary(leg)
                    orders_by_symbol[leg_summary["symbol"]].append(leg_summary)

        # Classify each position
        trailing_positions = []
        bracket_positions = []
        stop_only_positions = []
        unprotected_positions = []

        account = self.trading_client.get_account()
        equity = float(account.equity)

        print(f"\n  Account Equity: ${equity:,.2f}")
        print(f"  Positions: {len(positions)}")
        print(f"  Open Orders: {len(open_orders)}")

        print(f"\n  {'Symbol':8} {'Qty':>5} {'Entry':>8} {'Current':>8} "
              f"{'P&L%':>7} {'Protection':15} {'Details'}")
        print(f"  {'-' * 85}")

        for sym in sorted(positions.keys()):
            pos = positions[sym]
            sym_orders = orders_by_symbol.get(sym, [])

            # Classify protection
            has_trailing = False
            has_stop = False
            has_tp = False
            trail_pct = None
            trail_price = None
            stop_price = None
            tp_price = None
            hwm = None

            for o in sym_orders:
                if o["side"] != "sell":
                    continue
                if o["type"] == "trailing_stop":
                    has_trailing = True
                    trail_pct = o.get("trail_percent")
                    trail_price = o.get("trail_price")
                    hwm = o.get("hwm")
                elif o["type"] in ("stop", "stop_limit"):
                    has_stop = True
                    stop_price = o.get("stop_price")
                elif o["type"] == "limit" and o.get("order_class") in ("oco", "bracket"):
                    has_tp = True
                    tp_price = o.get("limit_price")

            # Build protection label and details
            if has_trailing:
                prot_type = "TRAILING STOP"
                details = f"Trail: {trail_pct}%"
                if hwm:
                    details += f" | HWM: ${hwm:.2f}"
                if trail_price:
                    details += f" | Stop: ${trail_price:.2f}"
                trailing_positions.append(sym)
            elif has_stop and has_tp:
                prot_type = "Bracket/OCO"
                details = f"SL: ${stop_price:.2f} | TP: ${tp_price:.2f}"
                bracket_positions.append(sym)
            elif has_stop:
                prot_type = "Stop only"
                details = f"SL: ${stop_price:.2f}"
                stop_only_positions.append(sym)
            else:
                prot_type = "UNPROTECTED"
                details = "!! NO PROTECTION !!"
                unprotected_positions.append(sym)

            pnl_pct = pos["unrealized_plpc"]
            pnl_icon = "+" if pnl_pct >= 0 else ""

            print(f"  {sym:8} {pos['qty']:>5.0f} ${pos['avg_entry_price']:>7.2f} "
                  f"${pos['current_price']:>7.2f} {pnl_icon}{pnl_pct:>5.1f}%  "
                  f"{prot_type:15} {details}")

        # Summary
        print(f"\n  {'=' * 70}")
        print(f"  PROTECTION SUMMARY")
        print(f"  {'=' * 70}")
        print(f"  Trailing stops:  {len(trailing_positions):>3}  {', '.join(trailing_positions) if trailing_positions else 'None'}")
        print(f"  Bracket/OCO:     {len(bracket_positions):>3}  {', '.join(bracket_positions) if bracket_positions else 'None'}")
        print(f"  Stop only:       {len(stop_only_positions):>3}  {', '.join(stop_only_positions) if stop_only_positions else 'None'}")
        print(f"  UNPROTECTED:     {len(unprotected_positions):>3}  {', '.join(unprotected_positions) if unprotected_positions else 'None'}")
        total = len(positions)
        protected = total - len(unprotected_positions)
        print(f"\n  Coverage: {protected}/{total} ({protected/total*100:.0f}%)")

        if trailing_positions:
            print(f"\n  TRAILING STOP DETAIL:")
            print(f"  {'Symbol':8} {'Entry':>8} {'Current':>8} {'P&L%':>7} "
                  f"{'Trail%':>7} {'Trail Stop':>10} {'Locked Gain':>12}")
            print(f"  {'-' * 75}")
            for sym in trailing_positions:
                pos = positions[sym]
                for o in orders_by_symbol.get(sym, []):
                    if o["type"] == "trailing_stop":
                        entry = pos["avg_entry_price"]
                        current = pos["current_price"]
                        pnl = pos["unrealized_plpc"]
                        tp = o.get("trail_percent", 0)

                        # Calculate effective stop
                        if o.get("trail_price"):
                            eff_stop = current - o["trail_price"]
                        elif o.get("hwm") and tp:
                            eff_stop = o["hwm"] * (1 - tp / 100)
                        else:
                            eff_stop = current * (1 - tp / 100)

                        locked = ((eff_stop / entry) - 1) * 100

                        print(f"  {sym:8} ${entry:>7.2f} ${current:>7.2f} "
                              f"{pnl:>+6.1f}%  {tp:>5.1f}%  "
                              f"${eff_stop:>9.2f}  {locked:>+10.1f}%")
                        break

        return {
            "trailing": trailing_positions,
            "bracket": bracket_positions,
            "stop_only": stop_only_positions,
            "unprotected": unprotected_positions,
        }

    def show_all_orders(self):
        """Show detailed breakdown of every open order."""
        print("\n" + "=" * 70)
        print("  ALL OPEN ORDERS (DETAILED)")
        print("=" * 70)

        orders = self.get_open_orders()
        if not orders:
            print("  No open orders.")
            return

        for o in sorted(orders, key=lambda x: x.symbol):
            s = self._order_summary(o)
            print(f"\n  {s['symbol']:8} {s['side'].upper():4} {s['qty']:>6.0f} "
                  f"Type: {s['type']:15} Class: {s['order_class']:10} "
                  f"Status: {s['status']}")

            if s['trail_percent']:
                print(f"           Trail: {s['trail_percent']}%"
                      + (f"  HWM: ${s['hwm']:.2f}" if s['hwm'] else "")
                      + (f"  Trail$: ${s['trail_price']:.2f}" if s['trail_price'] else ""))
            if s['stop_price']:
                print(f"           Stop: ${s['stop_price']:.2f}")
            if s['limit_price']:
                print(f"           Limit: ${s['limit_price']:.2f}")
            print(f"           Created: {s['created_at']}  ID: {s['id']}")

            # Show legs
            if o.legs:
                for leg in o.legs:
                    ls = self._order_summary(leg)
                    print(f"             Leg: {ls['type']:15} "
                          + (f"Stop: ${ls['stop_price']:.2f}" if ls['stop_price'] else "")
                          + (f"  Limit: ${ls['limit_price']:.2f}" if ls['limit_price'] else ""))

    def show_history(self):
        """Show today's filled and canceled orders — the trade history."""
        print("\n" + "=" * 70)
        print("  TODAY'S ORDER HISTORY (Filled + Canceled)")
        print("=" * 70)

        orders = self.get_todays_closed_orders()
        if not orders:
            print("  No closed orders found today.")
            return

        filled = []
        canceled = []
        other = []

        for o in orders:
            s = self._order_summary(o)
            status = s["status"]
            if status == "filled":
                filled.append(s)
            elif status == "canceled":
                canceled.append(s)
            else:
                other.append(s)

        if filled:
            print(f"\n  FILLED ORDERS ({len(filled)}):")
            print(f"  {'Symbol':8} {'Side':5} {'Qty':>6} {'Type':15} "
                  f"{'Fill Price':>10} {'Time':16}")
            print(f"  {'-' * 70}")
            for s in filled:
                fill_px = f"${s['filled_avg_price']:.2f}" if s['filled_avg_price'] else "N/A"
                print(f"  {s['symbol']:8} {s['side'].upper():5} {s['qty']:>6.0f} "
                      f"{s['type']:15} {fill_px:>10} {s['filled_at'] or 'N/A':16}")

        if canceled:
            print(f"\n  CANCELED ORDERS ({len(canceled)}):")
            print(f"  {'Symbol':8} {'Side':5} {'Qty':>6} {'Type':15} "
                  f"{'Class':10} {'Canceled At':16}")
            print(f"  {'-' * 70}")
            for s in canceled:
                print(f"  {s['symbol']:8} {s['side'].upper():5} {s['qty']:>6.0f} "
                      f"{s['type']:15} {s['order_class']:10} {s['canceled_at'] or 'N/A':16}")

        if other:
            print(f"\n  OTHER ({len(other)}):")
            for s in other:
                print(f"  {s['symbol']:8} {s['side']:5} Status: {s['status']}")

        # Also check execution log
        self._show_execution_log()

    def _show_execution_log(self):
        """Show the most recent execution log."""
        log_path = os.path.join(self.script_dir, "data", "last_execution.json")
        if not os.path.exists(log_path):
            return

        with open(log_path, "r") as f:
            log = json.load(f)

        print(f"\n  {'=' * 70}")
        print(f"  LAST EXECUTION LOG")
        print(f"  {'=' * 70}")
        print(f"  Timestamp: {log.get('timestamp', 'N/A')}")
        print(f"  Regime:    {log.get('regime', 'N/A')}")

        actions = log.get("actions", [])
        if actions:
            print(f"\n  Actions ({len(actions)}):")
            for a in actions:
                sym = a.get("symbol", "?")
                action = a.get("action", "?")
                status = a.get("status", "?")

                if action == "GRADUATE_TO_TRAILING":
                    print(f"    GRADUATED {sym}: Trail {a.get('trail_pct')}% "
                          f"| Stop ~${a.get('trail_stop_approx', 0):.2f} "
                          f"| Locks ~{a.get('locked_gain_pct', 0):+.1f}% "
                          f"| P&L at grad: {a.get('pnl_at_graduation', 0):+.1f}%")
                else:
                    side = a.get("side", "")
                    qty = a.get("qty", 0)
                    price = a.get("price", 0)
                    print(f"    {action:6} {side:4} {qty:>6} {sym:8} "
                          + (f"@ ${price:.2f}" if price else "")
                          + f"  [{status}]")

        graduation = log.get("graduation", {})
        if graduation:
            print(f"\n  Graduation: {graduation.get('graduated', 0)} graduated, "
                  f"{graduation.get('tightened', 0)} tightened, "
                  f"{graduation.get('skipped', 0)} skipped")

        errors = log.get("errors", [])
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    {e}")

        warnings = log.get("warnings", [])
        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    {w}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trailing Stop Verification Tool")
    parser.add_argument("--orders", action="store_true",
                        help="Show all open orders in detail")
    parser.add_argument("--history", action="store_true",
                        help="Show today's filled and canceled orders")
    args = parser.parse_args()

    verifier = TrailingStopVerifier()

    if args.orders:
        verifier.show_all_orders()
    elif args.history:
        verifier.show_history()
    else:
        # Default: full verification
        verifier.verify_all()
        print("")  # spacer
        verifier.show_history()

    return 0


if __name__ == "__main__":
    sys.exit(main())