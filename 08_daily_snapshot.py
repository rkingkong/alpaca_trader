#!/usr/bin/env python3
"""
Module 8: Daily Portfolio Snapshot & Order Reconciliation
==========================================================
Captures complete portfolio state daily and reconciles all orders.

Creates a structured daily folder:
  snapshots/
    2026-02-09/
      portfolio_summary.json       <- Account + positions + P&L
      open_orders.json             <- All currently open orders
      pending_sells.json           <- Sell orders not yet filled
      pending_buys.json            <- Buy orders not yet filled
      protection_status.json       <- Which positions have SL/TP
      order_history.json           <- All orders placed today
      reconciliation.json          <- What was planned vs. what happened
      execution_report.txt         <- Human-readable daily report

Run this module:
  - BEFORE rebalancing (to capture pre-trade state)
  - AFTER rebalancing (to capture post-trade state and reconcile)
  - Anytime to check current state

Usage:
  python 08_daily_snapshot.py                    # Full snapshot
  python 08_daily_snapshot.py --pre-trade        # Pre-trade snapshot
  python 08_daily_snapshot.py --post-trade       # Post-trade + reconciliation
  python 08_daily_snapshot.py --reconcile        # Reconcile planned vs actual
  python 08_daily_snapshot.py --cancel-stale     # Cancel unfilled limit orders
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import pytz

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # Not a reconfigurable stream

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


class DailyPortfolioSnapshot:
    """
    Captures and logs complete portfolio state with reconciliation.
    """

    def __init__(self, config_path: str = None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(self.script_dir, "config.json")

        self._load_config(config_path)
        self._setup_clients()

        # Create snapshots directory structure
        self.base_dir = os.path.join(self.script_dir, "snapshots")
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.today_dir = os.path.join(self.base_dir, self.today_str)
        os.makedirs(self.today_dir, exist_ok=True)

        # Also keep reference to data dir for recommendations
        self.data_dir = os.path.join(self.script_dir, "data")
        self.logs_dir = os.path.join(self.script_dir, "logs")

    def _load_config(self, config_path: str):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            self.alpaca_key = config.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY"))
            self.alpaca_secret = config.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY"))
            self.paper_trading = config.get("PAPER_TRADING", True)
        else:
            self.alpaca_key = os.getenv("ALPACA_API_KEY")
            self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
            self.paper_trading = True

    def _setup_clients(self):
        self.trading_client = TradingClient(
            self.alpaca_key, self.alpaca_secret, paper=self.paper_trading
        )
        self.stock_data_client = StockHistoricalDataClient(
            self.alpaca_key, self.alpaca_secret
        )

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def get_account(self) -> Dict:
        """Fetch account info."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "last_equity": float(account.last_equity),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "daily_pnl_pct": (
                ((float(account.equity) / float(account.last_equity)) - 1) * 100
                if float(account.last_equity) > 0
                else 0
            ),
            "status": account.status.value if hasattr(account.status, "value") else str(account.status),
            "trading_blocked": account.trading_blocked,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> Dict[str, Dict]:
        """Fetch all positions with full detail."""
        positions = self.trading_client.get_all_positions()
        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "qty_available": float(pos.qty_available) if hasattr(pos, "qty_available") and pos.qty_available else float(pos.qty),
                "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                "change_today": float(pos.change_today) * 100 if pos.change_today else 0,
                "asset_class": pos.asset_class.value if hasattr(pos.asset_class, "value") else str(pos.asset_class),
            }
        return result

    def get_all_orders(self, status: str = "open") -> List[Dict]:
        """Fetch orders by status: 'open', 'closed', 'all'."""
        if status == "open":
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        elif status == "closed":
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                nested=True,
                after=datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0),
            )
        else:
            request = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True)

        orders = self.trading_client.get_orders(filter=request)
        return [self._order_to_dict(o) for o in orders]

    def _order_to_dict(self, order) -> Dict:
        """Convert Alpaca order to dict with full detail."""
        result = {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id) if order.client_order_id else None,
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "type": order.type.value if hasattr(order.type, "value") else str(order.type),
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "order_class": (
                order.order_class.value
                if order.order_class and hasattr(order.order_class, "value")
                else str(order.order_class) if order.order_class else "simple"
            ),
            "time_in_force": order.time_in_force.value if hasattr(order.time_in_force, "value") else str(order.time_in_force),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "trail_percent": float(order.trail_percent) if order.trail_percent else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "expired_at": order.expired_at.isoformat() if order.expired_at else None,
            "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
            "legs": [self._order_to_dict(leg) for leg in order.legs] if order.legs else [],
        }
        return result

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze_protection_status(self, positions: Dict, orders: List[Dict]) -> Dict:
        """
        Analyze which positions have proper protection (stop-loss + take-profit).
        Returns detailed protection status for each position.
        """
        protection = {}

        # Build order map per symbol
        symbol_orders = defaultdict(list)
        for order in orders:
            if order["side"] == "sell" and order["status"] in ("new", "accepted", "held", "partially_filled"):
                symbol_orders[order["symbol"]].append(order)

        for symbol, pos in positions.items():
            info = {
                "symbol": symbol,
                "qty": pos["qty"],
                "entry_price": pos["avg_entry_price"],
                "current_price": pos["current_price"],
                "unrealized_plpc": pos["unrealized_plpc"],
                "has_stop_loss": False,
                "has_take_profit": False,
                "has_trailing_stop": False,
                "stop_loss_price": None,
                "take_profit_price": None,
                "trail_percent": None,
                "protection_type": "NONE",
                "order_class": "none",
                "protection_orders": [],
                "risk_pct": None,
                "reward_pct": None,
            }

            for order in symbol_orders.get(symbol, []):
                otype = order["type"]
                oclass = order.get("order_class", "simple")
                info["protection_orders"].append({
                    "id": order["id"],
                    "type": otype,
                    "class": oclass,
                    "limit_price": order.get("limit_price"),
                    "stop_price": order.get("stop_price"),
                    "trail_percent": order.get("trail_percent"),
                    "qty": order.get("qty"),
                    "status": order["status"],
                })

                if otype in ("stop", "stop_limit"):
                    info["has_stop_loss"] = True
                    info["stop_loss_price"] = order.get("stop_price")
                elif otype == "trailing_stop":
                    info["has_trailing_stop"] = True
                    info["trail_percent"] = order.get("trail_percent")
                elif otype == "limit" and oclass in ("oco", "bracket"):
                    info["has_take_profit"] = True
                    info["take_profit_price"] = order.get("limit_price")
                elif otype == "limit" and oclass == "simple":
                    # Simple limit sell - NOT real protection, this is just a limit order
                    info["has_take_profit"] = True
                    info["take_profit_price"] = order.get("limit_price")

                # Check legs for bracket/OCO
                for leg in order.get("legs", []):
                    leg_type = leg.get("type", "")
                    if leg_type in ("stop", "stop_limit"):
                        info["has_stop_loss"] = True
                        info["stop_loss_price"] = leg.get("stop_price")
                    elif leg_type == "limit":
                        info["has_take_profit"] = True
                        info["take_profit_price"] = leg.get("limit_price")

            # Determine protection type
            if info["has_trailing_stop"]:
                info["protection_type"] = "TRAILING_STOP"
            elif info["has_stop_loss"] and info["has_take_profit"]:
                info["protection_type"] = "FULL_OCO"
            elif info["has_stop_loss"]:
                info["protection_type"] = "STOP_ONLY"
            elif info["has_take_profit"]:
                info["protection_type"] = "TP_ONLY_VULNERABLE"
            else:
                info["protection_type"] = "NONE"

            # Calculate risk/reward
            entry = pos["avg_entry_price"]
            if info["stop_loss_price"] and entry > 0:
                info["risk_pct"] = ((info["stop_loss_price"] - entry) / entry) * 100
            if info["take_profit_price"] and entry > 0:
                info["reward_pct"] = ((info["take_profit_price"] - entry) / entry) * 100

            protection[symbol] = info

        return protection

    def categorize_orders(self, orders: List[Dict]) -> Dict:
        """Categorize all open orders into meaningful groups."""
        categories = {
            "pending_sells": [],
            "pending_buys": [],
            "stop_losses": [],
            "take_profits": [],
            "trailing_stops": [],
            "bracket_orders": [],
            "oco_orders": [],
            "simple_limits": [],
            "market_orders": [],
            "stale_orders": [],  # Orders older than 24h that haven't filled
        }

        now = datetime.now(pytz.UTC)
        for order in orders:
            otype = order["type"]
            oclass = order.get("order_class", "simple")
            oside = order["side"]
            status = order["status"]

            if status not in ("new", "accepted", "held", "partially_filled", "pending_new"):
                continue

            # Check if stale (older than 24h)
            created = order.get("created_at")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if (now - created_dt).total_seconds() > 86400:
                        categories["stale_orders"].append(order)
                except Exception:
                    pass

            # Categorize
            if oclass == "bracket":
                categories["bracket_orders"].append(order)
            elif oclass == "oco":
                categories["oco_orders"].append(order)

            if otype == "trailing_stop":
                categories["trailing_stops"].append(order)
            elif otype in ("stop", "stop_limit"):
                categories["stop_losses"].append(order)
            elif otype == "limit" and oclass in ("oco", "bracket"):
                categories["take_profits"].append(order)
            elif otype == "limit" and oclass == "simple":
                categories["simple_limits"].append(order)
                if oside == "sell":
                    categories["pending_sells"].append(order)
                else:
                    categories["pending_buys"].append(order)
            elif otype == "market":
                categories["market_orders"].append(order)
                if oside == "sell":
                    categories["pending_sells"].append(order)
                else:
                    categories["pending_buys"].append(order)

        return categories

    def identify_problem_orders(self, orders: List[Dict], positions: Dict) -> List[Dict]:
        """
        Identify orders that are likely problematic:
        - Limit sells far from market price (won't fill)
        - Simple limit sells that should be brackets
        - Sell orders for positions we don't hold
        - Stale unfilled orders
        """
        problems = []
        now = datetime.now(pytz.UTC)

        for order in orders:
            if order["status"] not in ("new", "accepted", "held"):
                continue

            symbol = order["symbol"]
            otype = order["type"]
            oclass = order.get("order_class", "simple")
            oside = order["side"]
            limit_price = order.get("limit_price")

            # Problem 1: Simple limit sell (not OCO/bracket) — no stop-loss protection
            if oside == "sell" and otype == "limit" and oclass == "simple":
                pos = positions.get(symbol)
                if pos:
                    current_price = pos["current_price"]
                    if limit_price and current_price > 0:
                        distance_pct = ((limit_price - current_price) / current_price) * 100
                        problems.append({
                            "order_id": order["id"],
                            "symbol": symbol,
                            "type": "SIMPLE_LIMIT_SELL",
                            "severity": "HIGH",
                            "detail": (
                                f"Simple limit sell at ${limit_price:.2f} "
                                f"({distance_pct:+.1f}% from current ${current_price:.2f}). "
                                f"This is NOT a bracket/OCO — has NO stop-loss protection. "
                                f"Position is NAKED on the downside."
                            ),
                            "recommendation": "Cancel and replace with OCO (stop-loss + take-profit) or bracket order",
                        })

            # Problem 2: Sell order for symbol not in positions
            if oside == "sell" and symbol not in positions:
                problems.append({
                    "order_id": order["id"],
                    "symbol": symbol,
                    "type": "ORPHAN_SELL",
                    "severity": "MEDIUM",
                    "detail": f"Sell order for {symbol} but no position held. Orphan order.",
                    "recommendation": "Cancel this order",
                })

            # Problem 3: Stale orders (>24h old)
            created = order.get("created_at")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age_hours = (now - created_dt).total_seconds() / 3600
                    if age_hours > 24 and otype not in ("stop", "stop_limit", "trailing_stop"):
                        problems.append({
                            "order_id": order["id"],
                            "symbol": symbol,
                            "type": "STALE_ORDER",
                            "severity": "LOW",
                            "detail": f"Order is {age_hours:.0f} hours old and unfilled. Type: {otype} {oclass}.",
                            "recommendation": "Review and cancel if no longer needed",
                        })
                except Exception:
                    pass

        return problems

    # =========================================================================
    # SNAPSHOT CREATION
    # =========================================================================

    def capture_full_snapshot(self, label: str = "snapshot") -> Dict:
        """Capture complete portfolio state and save to daily folder."""
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*70}")
        print(f"  DAILY PORTFOLIO SNAPSHOT — {label.upper()}")
        print(f"{'='*70}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Folder: {self.today_dir}")

        # 1. Account
        print("\n  Fetching account info...")
        account = self.get_account()
        print(f"    Equity: ${account['equity']:,.2f}")
        print(f"    Cash: ${account['cash']:,.2f}")
        print(f"    Daily P&L: {account['daily_pnl_pct']:+.2f}% (${account['daily_pnl']:+,.2f})")

        # 2. Positions
        print("\n  Fetching positions...")
        positions = self.get_positions()
        print(f"    Total positions: {len(positions)}")

        total_value = sum(p["market_value"] for p in positions.values())
        total_pnl = sum(p["unrealized_pl"] for p in positions.values())
        print(f"    Total market value: ${total_value:,.2f}")
        print(f"    Total unrealized P&L: ${total_pnl:+,.2f}")

        # 3. Open orders
        print("\n  Fetching open orders...")
        open_orders = self.get_all_orders("open")
        print(f"    Open orders: {len(open_orders)}")

        # 4. Today's closed orders
        print("  Fetching today's closed orders...")
        try:
            closed_orders = self.get_all_orders("closed")
        except Exception:
            closed_orders = []
        print(f"    Closed today: {len(closed_orders)}")

        # 5. Categorize orders
        categories = self.categorize_orders(open_orders)

        # 6. Protection analysis
        print("\n  Analyzing protection status...")
        protection = self.analyze_protection_status(positions, open_orders)

        protected_count = sum(1 for p in protection.values() if p["protection_type"] in ("FULL_OCO", "TRAILING_STOP"))
        stop_only = sum(1 for p in protection.values() if p["protection_type"] == "STOP_ONLY")
        tp_only = sum(1 for p in protection.values() if p["protection_type"] == "TP_ONLY_VULNERABLE")
        naked = sum(1 for p in protection.values() if p["protection_type"] == "NONE")

        print(f"    Fully protected (SL+TP):  {protected_count}")
        print(f"    Stop-loss only:           {stop_only}")
        print(f"    Take-profit only (RISKY): {tp_only}")
        print(f"    Completely naked:         {naked}")

        # 7. Problem identification
        problems = self.identify_problem_orders(open_orders, positions)
        if problems:
            print(f"\n  ⚠️  PROBLEMS DETECTED: {len(problems)}")
            for p in problems:
                print(f"    [{p['severity']}] {p['symbol']}: {p['type']}")
                print(f"         {p['detail']}")

        # 8. Save all files
        print("\n  Saving snapshot files...")

        # Portfolio summary
        summary = {
            "timestamp": timestamp,
            "label": label,
            "account": account,
            "positions_count": len(positions),
            "total_market_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "total_cost_basis": sum(p.get("cost_basis", 0) for p in positions.values()),
            "open_orders_count": len(open_orders),
            "protection_summary": {
                "fully_protected": protected_count,
                "stop_only": stop_only,
                "tp_only_vulnerable": tp_only,
                "naked": naked,
            },
            "problems_count": len(problems),
            "positions": positions,
        }
        self._save_json("portfolio_summary.json", summary)

        # Open orders
        self._save_json("open_orders.json", {
            "timestamp": timestamp,
            "total": len(open_orders),
            "categories": {k: len(v) for k, v in categories.items()},
            "orders": open_orders,
        })

        # Pending sells
        pending_sells = [
            o for o in open_orders
            if o["side"] == "sell"
            and o["status"] in ("new", "accepted", "held", "partially_filled")
            and o.get("order_class", "simple") == "simple"
            and o["type"] == "limit"
        ]
        self._save_json("pending_sells.json", {
            "timestamp": timestamp,
            "count": len(pending_sells),
            "note": "These are simple limit sells — NOT part of OCO/bracket protection",
            "orders": pending_sells,
        })

        # Pending buys
        pending_buys = [
            o for o in open_orders
            if o["side"] == "buy"
            and o["status"] in ("new", "accepted", "held", "partially_filled")
        ]
        self._save_json("pending_buys.json", {
            "timestamp": timestamp,
            "count": len(pending_buys),
            "orders": pending_buys,
        })

        # Protection status
        self._save_json("protection_status.json", {
            "timestamp": timestamp,
            "summary": {
                "fully_protected": protected_count,
                "stop_only": stop_only,
                "tp_only_vulnerable": tp_only,
                "naked": naked,
            },
            "positions": protection,
        })

        # Order history (today's closed)
        self._save_json("order_history.json", {
            "timestamp": timestamp,
            "closed_today": len(closed_orders),
            "orders": closed_orders,
        })

        # Problems
        self._save_json("problems.json", {
            "timestamp": timestamp,
            "count": len(problems),
            "problems": problems,
        })

        # Human-readable report
        self._write_execution_report(
            account, positions, open_orders, categories, protection, problems, label
        )

        print(f"\n  ✅ Snapshot saved to: {self.today_dir}")
        return {
            "success": True,
            "snapshot_dir": self.today_dir,
            "account": account,
            "positions": positions,
            "open_orders": open_orders,
            "protection": protection,
            "problems": problems,
            "categories": categories,
        }

    def _save_json(self, filename: str, data: Dict):
        """Save JSON to today's snapshot folder."""
        path = os.path.join(self.today_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    def _write_execution_report(
        self, account, positions, open_orders, categories, protection, problems, label
    ):
        """Write human-readable daily report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  APEX QUANT CAPITAL — DAILY PORTFOLIO REPORT ({label.upper()})")
        lines.append(f"  Date: {self.today_str}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # Account
        lines.append("\n  ACCOUNT OVERVIEW")
        lines.append(f"  {'─'*50}")
        lines.append(f"  Equity:        ${account['equity']:>12,.2f}")
        lines.append(f"  Cash:          ${account['cash']:>12,.2f}")
        lines.append(f"  Buying Power:  ${account['buying_power']:>12,.2f}")
        lines.append(f"  Daily P&L:     ${account['daily_pnl']:>+12,.2f} ({account['daily_pnl_pct']:+.2f}%)")

        # Positions
        lines.append(f"\n  POSITIONS ({len(positions)} total)")
        lines.append(f"  {'─'*50}")
        lines.append(f"  {'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8} {'Protection':<20}")

        sorted_pos = sorted(positions.values(), key=lambda x: x["market_value"], reverse=True)
        for pos in sorted_pos:
            symbol = pos["symbol"]
            prot = protection.get(symbol, {})
            prot_type = prot.get("protection_type", "UNKNOWN")
            prot_icon = {
                "FULL_OCO": "[OK] OCO",
                "TRAILING_STOP": "[OK] Trail",
                "STOP_ONLY": "[!!] SL only",
                "TP_ONLY_VULNERABLE": "[XX] TP only!",
                "NONE": "[XX] NAKED!",
            }.get(prot_type, "[??]")

            lines.append(
                f"  {symbol:<8} {pos['qty']:>6.0f} "
                f"${pos['avg_entry_price']:>9.2f} "
                f"${pos['current_price']:>9.2f} "
                f"${pos['unrealized_pl']:>+9.2f} "
                f"{pos['unrealized_plpc']:>+7.1f}% "
                f"{prot_icon}"
            )

        # Open Orders Summary
        lines.append(f"\n  OPEN ORDERS ({len(open_orders)} total)")
        lines.append(f"  {'─'*50}")
        lines.append(f"  Bracket orders:     {len(categories['bracket_orders'])}")
        lines.append(f"  OCO orders:         {len(categories['oco_orders'])}")
        lines.append(f"  Trailing stops:     {len(categories['trailing_stops'])}")
        lines.append(f"  Stop losses:        {len(categories['stop_losses'])}")
        lines.append(f"  Simple limit sells: {len(categories['simple_limits'])}")
        lines.append(f"  Stale (>24h):       {len(categories['stale_orders'])}")

        # Simple limit sells detail (these are the problematic ones)
        simple_sells = [o for o in open_orders if o["side"] == "sell" and o["type"] == "limit" and o.get("order_class", "simple") == "simple"]
        if simple_sells:
            lines.append(f"\n  [!!] SIMPLE LIMIT SELLS (not bracket/OCO -- may not fill):")
            for o in simple_sells:
                filled = o.get("filled_qty", 0)
                remaining = (o.get("qty", 0) or 0) - filled
                lines.append(
                    f"    {o['symbol']:<8} Limit @ ${o.get('limit_price', 0):>10.2f} "
                    f"Qty: {remaining:.0f}  Status: {o['status']}"
                )

        # Problems
        if problems:
            lines.append(f"\n  ⚠️  PROBLEMS ({len(problems)})")
            lines.append(f"  {'─'*50}")
            for p in problems:
                lines.append(f"  [{p['severity']}] {p['symbol']}: {p['type']}")
                lines.append(f"    {p['detail']}")
                lines.append(f"    → {p['recommendation']}")

        # Protection summary
        prot_summary = {
            "FULL_OCO": [], "TRAILING_STOP": [], "STOP_ONLY": [],
            "TP_ONLY_VULNERABLE": [], "NONE": []
        }
        for symbol, info in protection.items():
            prot_summary[info["protection_type"]].append(symbol)

        lines.append(f"\n  PROTECTION STATUS")
        lines.append(f"  {'─'*50}")
        if prot_summary["NONE"]:
            lines.append(f"  ❌ NAKED (no protection):       {', '.join(prot_summary['NONE'])}")
        if prot_summary["TP_ONLY_VULNERABLE"]:
            lines.append(f"  ❌ TP-only (no stop-loss):      {', '.join(prot_summary['TP_ONLY_VULNERABLE'])}")
        if prot_summary["STOP_ONLY"]:
            lines.append(f"  ⚠️  Stop-loss only:             {', '.join(prot_summary['STOP_ONLY'])}")
        if prot_summary["FULL_OCO"]:
            lines.append(f"  ✅ Full OCO:                    {', '.join(prot_summary['FULL_OCO'])}")
        if prot_summary["TRAILING_STOP"]:
            lines.append(f"  ✅ Trailing stop:               {', '.join(prot_summary['TRAILING_STOP'])}")

        lines.append(f"\n{'='*70}")

        report_text = "\n".join(lines)
        report_path = os.path.join(self.today_dir, f"execution_report_{label}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Also print to console (safe for Windows)
        try:
            print(report_text)
        except UnicodeEncodeError:
            print(report_text.encode("ascii", errors="replace").decode("ascii"))

    # =========================================================================
    # RECONCILIATION
    # =========================================================================

    def reconcile(self) -> Dict:
        """
        Compare what was planned (from recommendations.json) vs what actually
        happened (from order history and current positions).
        """
        print(f"\n{'='*70}")
        print(f"  ORDER RECONCILIATION")
        print(f"{'='*70}")

        # Load recommendations
        rec_path = os.path.join(self.data_dir, "recommendations.json")
        if not os.path.exists(rec_path):
            print("  No recommendations.json found — skipping reconciliation")
            return {"success": False, "error": "No recommendations"}

        with open(rec_path, "r") as f:
            recs = json.load(f)

        buy_signals = {s["symbol"]: s for s in recs.get("buy_signals", [])}
        sell_signals = {s["symbol"]: s for s in recs.get("sell_signals", [])}

        # Current state
        positions = self.get_positions()
        open_orders = self.get_all_orders("open")

        # Track results
        reconciliation = {
            "timestamp": datetime.now().isoformat(),
            "planned_buys": [],
            "planned_sells": [],
            "executed_buys": [],
            "executed_sells": [],
            "pending_orders": [],
            "failed_or_missing": [],
        }

        # Check buy signals
        print(f"\n  BUY SIGNALS ({len(buy_signals)}):")
        for symbol, signal in buy_signals.items():
            status = "UNKNOWN"
            if symbol in positions:
                status = "EXECUTED (position exists)"
                reconciliation["executed_buys"].append(symbol)
            else:
                # Check if there's a pending buy order
                pending = [o for o in open_orders if o["symbol"] == symbol and o["side"] == "buy"]
                if pending:
                    status = f"PENDING ({pending[0]['status']})"
                    reconciliation["pending_orders"].append({"symbol": symbol, "direction": "buy"})
                else:
                    status = "NOT EXECUTED"
                    reconciliation["failed_or_missing"].append({"symbol": symbol, "direction": "buy"})

            reconciliation["planned_buys"].append({
                "symbol": symbol,
                "confidence": signal.get("confidence", 0),
                "status": status,
            })
            icon = "✅" if "EXECUTED" in status else "⏳" if "PENDING" in status else "❌"
            print(f"    {icon} {symbol:<8} Conf: {signal.get('confidence', 0):.0%} → {status}")

        # Check sell signals
        print(f"\n  SELL SIGNALS ({len(sell_signals)}):")
        for symbol, signal in sell_signals.items():
            status = "UNKNOWN"
            if symbol not in positions:
                status = "EXECUTED (position closed)"
                reconciliation["executed_sells"].append(symbol)
            else:
                # Check if there's a pending sell order
                pending = [o for o in open_orders if o["symbol"] == symbol and o["side"] == "sell"]
                if pending:
                    order_types = set(o["type"] for o in pending)
                    order_classes = set(o.get("order_class", "simple") for o in pending)
                    status = f"PENDING ({', '.join(order_types)}) class=({', '.join(order_classes)})"
                    reconciliation["pending_orders"].append({"symbol": symbol, "direction": "sell"})
                else:
                    status = "NOT EXECUTED (still holding)"
                    reconciliation["failed_or_missing"].append({"symbol": symbol, "direction": "sell"})

            reconciliation["planned_sells"].append({
                "symbol": symbol,
                "confidence": signal.get("confidence", 0),
                "status": status,
            })
            icon = "✅" if "EXECUTED" in status else "⏳" if "PENDING" in status else "❌"
            print(f"    {icon} {symbol:<8} Conf: {signal.get('confidence', 0):.0%} → {status}")

        # Summary
        print(f"\n  RECONCILIATION SUMMARY:")
        print(f"    Buys executed:  {len(reconciliation['executed_buys'])}/{len(buy_signals)}")
        print(f"    Sells executed: {len(reconciliation['executed_sells'])}/{len(sell_signals)}")
        print(f"    Still pending:  {len(reconciliation['pending_orders'])}")
        print(f"    Failed/Missing: {len(reconciliation['failed_or_missing'])}")

        self._save_json("reconciliation.json", reconciliation)
        return reconciliation

    # =========================================================================
    # STALE ORDER CLEANUP
    # =========================================================================

    def cancel_stale_limit_sells(self, dry_run: bool = True) -> Dict:
        """
        Cancel simple limit sell orders that are sitting unfilled.
        These are the orders visible in the screenshot — limit sells placed as
        exit orders but not filling because price hasn't reached the limit.
        
        These should either be:
        1. Market sells (if we want to exit now)
        2. Part of OCO/bracket (if they're take-profit levels)
        """
        print(f"\n{'='*70}")
        print(f"  STALE LIMIT SELL CLEANUP {'(DRY RUN)' if dry_run else '(LIVE)'}")
        print(f"{'='*70}")

        open_orders = self.get_all_orders("open")
        positions = self.get_positions()

        # Find simple limit sells
        stale_sells = [
            o for o in open_orders
            if o["side"] == "sell"
            and o["type"] == "limit"
            and o.get("order_class", "simple") == "simple"
            and o["status"] in ("new", "accepted", "held")
            and o.get("filled_qty", 0) == 0
        ]

        if not stale_sells:
            print("  No stale limit sells found. All clear!")
            return {"canceled": 0}

        print(f"\n  Found {len(stale_sells)} simple limit sell orders:")
        canceled = 0

        for order in stale_sells:
            symbol = order["symbol"]
            limit_price = order.get("limit_price", 0)
            qty = order.get("qty", 0)
            pos = positions.get(symbol)
            current_price = pos["current_price"] if pos else 0

            distance = ""
            if current_price and limit_price:
                dist_pct = ((limit_price - current_price) / current_price) * 100
                distance = f" ({dist_pct:+.1f}% from market)"

            print(f"    {symbol:<8} Limit @ ${limit_price:.2f}{distance}  Qty: {qty}")

            if dry_run:
                print(f"      [DRY RUN] Would cancel")
            else:
                try:
                    self.trading_client.cancel_order_by_id(order["id"])
                    print(f"      ✅ Canceled")
                    canceled += 1
                    time.sleep(0.3)
                except Exception as e:
                    print(f"      ❌ Failed: {e}")

        result = {"canceled": canceled if not dry_run else 0, "would_cancel": len(stale_sells)}
        self._save_json("stale_cleanup.json", {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            **result,
            "orders": stale_sells,
        })

        return result

    # =========================================================================
    # HISTORY BROWSER
    # =========================================================================

    def list_snapshots(self) -> List[str]:
        """List all available snapshot dates."""
        if not os.path.exists(self.base_dir):
            return []
        dates = sorted(
            [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))],
            reverse=True,
        )
        return dates

    def load_snapshot(self, date_str: str, filename: str) -> Optional[Dict]:
        """Load a specific snapshot file from a given date."""
        path = os.path.join(self.base_dir, date_str, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily Portfolio Snapshot & Reconciliation")
    parser.add_argument("--pre-trade", action="store_true", help="Take pre-trade snapshot")
    parser.add_argument("--post-trade", action="store_true", help="Take post-trade snapshot + reconcile")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile planned vs actual")
    parser.add_argument("--cancel-stale", action="store_true", help="Cancel stale unfilled limit sells")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for cancel operations")
    parser.add_argument("--history", action="store_true", help="List available snapshot dates")

    args = parser.parse_args()
    snapshot = DailyPortfolioSnapshot()

    if args.history:
        dates = snapshot.list_snapshots()
        print(f"\nAvailable snapshots ({len(dates)}):")
        for d in dates[:30]:
            print(f"  {d}")
        return 0

    if args.pre_trade:
        snapshot.capture_full_snapshot(label="pre_trade")
    elif args.post_trade:
        snapshot.capture_full_snapshot(label="post_trade")
        snapshot.reconcile()
    elif args.reconcile:
        snapshot.reconcile()
    elif args.cancel_stale:
        snapshot.cancel_stale_limit_sells(dry_run=args.dry_run)
    else:
        # Default: full snapshot
        snapshot.capture_full_snapshot(label="snapshot")
        snapshot.reconcile()

    return 0


if __name__ == "__main__":
    exit(main())