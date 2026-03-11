#!/usr/bin/env python3
"""
Module 2: Portfolio Status
===========================
Fetches complete portfolio state from Alpaca:
  - Account info (equity, cash, buying power, daily P&L)
  - All open positions (stocks, crypto, options)
  - All open orders
  - Risk metrics (concentration, exposure)

Outputs: data/portfolio_status.json (consumed by execution engine + options)
"""

import os
import sys
import re
from datetime import datetime, timedelta
from typing import Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

try:
    from config import Config, load_regime_context, DATA_DIR, save_json, print_header
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config, load_regime_context, DATA_DIR, save_json, print_header


class PortfolioStatus:
    """Fetches and reports complete portfolio state."""

    def __init__(self):
        cfg = Config()
        self.paper_trading = cfg.PAPER_TRADING
        self.client = TradingClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, paper=cfg.PAPER_TRADING
        )

    def fetch(self) -> Dict:
        """Fetch complete portfolio status and save to file."""
        print_header("PORTFOLIO STATUS")
        print(f"  Fetching: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        account = self._get_account()
        positions = self._get_positions()
        open_orders = self._get_open_orders()
        recent_orders = self._get_recent_orders()
        risk = self._calc_risk(account, positions)

        # Load regime context
        regime = {}
        ctx = load_regime_context()
        if ctx:
            regime = {"regime": ctx.regime, "regime_score": ctx.regime_score}
            print(f"  Regime: {ctx.regime.upper()} (score: {ctx.regime_score:+.3f})")

        status = {
            "fetch_timestamp": datetime.now().isoformat(),
            "paper_trading": self.paper_trading,
            "regime": regime,
            "account": account,
            "positions": positions,
            "open_orders": open_orders,
            "recent_orders": recent_orders,
            "risk_metrics": risk,
            "summary": {
                "equity": account.get("equity", 0),
                "cash": account.get("cash", 0),
                "buying_power": account.get("buying_power", 0),
                "daily_pnl": account.get("daily_pnl", 0),
                "daily_pnl_pct": account.get("daily_pnl_pct", 0),
                "total_positions": len([p for p in positions if "error" not in p]),
                "stock_positions": len([p for p in positions if not p.get("is_option")]),
                "option_positions": len([p for p in positions if p.get("is_option")]),
                "open_orders_count": len(open_orders),
            }
        }

        save_json("portfolio_status.json", status)
        self._print_summary(account, positions, open_orders, risk)
        return status

    # â”€â”€â”€ DATA FETCHING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _get_account(self) -> Dict:
        try:
            a = self.client.get_account()
            equity = float(a.equity)
            last_eq = float(a.last_equity)
            return {
                "status": a.status.value if hasattr(a.status, 'value') else str(a.status),
                "equity": equity,
                "cash": float(a.cash),
                "last_equity": last_eq,
                "daily_pnl": round(equity - last_eq, 2),
                "daily_pnl_pct": round(((equity / last_eq) - 1) * 100, 2) if last_eq > 0 else 0,
                "buying_power": float(a.buying_power),
                "portfolio_value": float(a.portfolio_value),
                "pattern_day_trader": a.pattern_day_trader,
                "trading_blocked": a.trading_blocked,
                "daytrade_count": a.daytrade_count,
                "options_trading_level": getattr(a, 'options_trading_level', None),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_positions(self, max_retries: int = 3) -> List[Dict]:
        """Fetch all positions with retry logic for transient API failures."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                positions = self.client.get_all_positions()
                result = []
                for p in positions:
                    sym = p.symbol
                    is_option = self._is_option(sym) or (
                        hasattr(p.asset_class, 'value') and p.asset_class.value == "us_option"
                    )
                    result.append({
                        "symbol": sym,
                        "asset_class": p.asset_class.value if hasattr(p.asset_class, 'value') else str(p.asset_class),
                        "is_option": is_option,
                        "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
                        "qty": float(p.qty),
                        "qty_available": float(p.qty_available) if p.qty_available else float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "market_value": float(p.market_value),
                        "cost_basis": float(p.cost_basis),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc) * 100,
                        "change_today": float(p.change_today) * 100 if p.change_today else 0,
                    })
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt  # Exponential backoff: 2s, 4s
                    print(f"  \u26a0\ufe0f  Positions API failed (attempt {attempt}/{max_retries}): {e}")
                    print(f"      Retrying in {wait}s...")
                    import time
                    time.sleep(wait)
        print(f"  \u274c Positions API failed after {max_retries} attempts: {last_error}")
        return [{"error": str(last_error)}]

    def _get_open_orders(self) -> List[Dict]:
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
            orders = self.client.get_orders(filter=request)
            return [self._order_dict(o) for o in orders]
        except Exception as e:
            return [{"error": str(e)}]

    def _get_recent_orders(self, limit: int = 20) -> List[Dict]:
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit,
                nested=True,
                after=datetime.now() - timedelta(days=3),
            )
            orders = self.client.get_orders(filter=request)
            return [self._order_dict(o) for o in orders]
        except Exception:
            return []

    # â”€â”€â”€ RISK METRICS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _calc_risk(self, account: Dict, positions: List[Dict]) -> Dict:
        equity = account.get("equity", 0)
        if equity <= 0:
            return {}

        allocations = []
        for p in positions:
            if "error" in p:
                continue
            alloc = abs(p["market_value"]) / equity * 100
            allocations.append({
                "symbol": p["symbol"],
                "allocation_pct": round(alloc, 2),
                "pnl_pct": round(p.get("unrealized_plpc", 0), 2),
                "is_option": p.get("is_option", False),
            })

        allocations.sort(key=lambda x: x["allocation_pct"], reverse=True)
        total_value = sum(abs(p.get("market_value", 0)) for p in positions if "error" not in p)
        total_pnl = sum(p.get("unrealized_pl", 0) for p in positions if "error" not in p)
        over_concentrated = [a for a in allocations if a["allocation_pct"] > 5 and not a["is_option"]]

        return {
            "total_positions": len([p for p in positions if "error" not in p]),
            "total_market_value": round(total_value, 2),
            "total_unrealized_pnl": round(total_pnl, 2),
            "cash_pct": round(account.get("cash", 0) / equity * 100, 2),
            "invested_pct": round(total_value / equity * 100, 2),
            "allocations": allocations,
            "over_concentrated": over_concentrated,
            "max_allocation": allocations[0]["allocation_pct"] if allocations else 0,
        }

    # â”€â”€â”€ HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _order_dict(self, order) -> Dict:
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else None,
            "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
            "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "order_class": (order.order_class.value if order.order_class and
                            hasattr(order.order_class, 'value') else "simple"),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "legs": [self._order_dict(leg) for leg in order.legs] if order.legs else [],
        }

    @staticmethod
    def _is_option(symbol: str) -> bool:
        if not symbol or len(symbol) < 10:
            return False
        return bool(re.match(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$', symbol))

    def _print_summary(self, account, positions, orders, risk):
        print(f"\n  {'=' * 60}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"  {'=' * 60}")

        if "error" not in account:
            print(f"  Equity:      ${account['equity']:,.2f}")
            print(f"  Cash:        ${account['cash']:,.2f}")
            print(f"  Buying Power: ${account['buying_power']:,.2f}")
            pnl = account['daily_pnl']
            pnl_pct = account['daily_pnl_pct']
            icon = "ðŸ“ˆ" if pnl >= 0 else "ðŸ“‰"
            print(f"  Daily P&L:   {icon} ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        valid_pos = [p for p in positions if "error" not in p]
        print(f"\n  Positions: {len(valid_pos)}")
        for p in valid_pos[:8]:
            pnl_icon = "+" if p['unrealized_plpc'] >= 0 else ""
            print(f"    {p['symbol']:8} {p['qty']:>6} @ ${p['current_price']:.2f} "
                  f"({pnl_icon}{p['unrealized_plpc']:.1f}%)")
        if len(valid_pos) > 8:
            print(f"    ... and {len(valid_pos) - 8} more")

        print(f"\n  Open Orders: {len([o for o in orders if 'error' not in o])}")

        if risk.get("over_concentrated"):
            print(f"\n  âš ï¸  OVER-CONCENTRATED (>5%):")
            for a in risk["over_concentrated"]:
                print(f"    {a['symbol']}: {a['allocation_pct']:.1f}%")

        print(f"\n  Results saved to: data/portfolio_status.json")


def main():
    try:
        ps = PortfolioStatus()
        status = ps.fetch()

        # Risk assessment
        pnl_pct = status.get("account", {}).get("daily_pnl_pct", 0)
        if pnl_pct <= -2:
            print("\n  ðŸ”´ WARNING: Daily loss exceeds 2%")
        elif pnl_pct <= -1:
            print("\n  ðŸŸ¡ CAUTION: Daily loss approaching 1%")
        else:
            print("\n  ðŸŸ¢ Daily P&L within acceptable range")

        return 0
    except Exception as e:
        print(f"\n  âŒ Portfolio status failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())