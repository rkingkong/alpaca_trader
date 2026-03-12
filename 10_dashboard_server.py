#!/usr/bin/env python3
"""
Module 10: Dashboard Web Server
=================================
Lightweight HTTP server that serves the Luxverum Capital trading terminal.
Reads from existing data files and serves JSON API endpoints for the
React dashboard to consume.

Endpoints:
  GET /api/status          — Full dashboard data export
  GET /api/positions       — Current positions
  GET /api/signals         — Latest AI signals
  GET /api/news            — News & market intelligence
  GET /api/performance     — Equity history & performance metrics
  GET /api/regime          — Current market regime
  GET /api/protection      — Protection audit status
  GET /api/earnings        — Upcoming earnings calendar
  GET /api/economic        — Economic indicators
  GET /api/sentiment       — News sentiment summary
  GET /api/snapshot/:date  — Historical snapshot
  GET /                    — Serves the dashboard HTML

Usage:
  python 10_dashboard_server.py                    # Start on port 8080
  python 10_dashboard_server.py --port 3000        # Custom port
  python 10_dashboard_server.py --host 0.0.0.0     # Allow external access
  python 10_dashboard_server.py --refresh           # Collect news before starting
"""

import os
import sys
import json
import argparse
import mimetypes
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Optional

try:
    from config import Config, print_header
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config, print_header


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """Loads data from the trading system's data directory."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, "data")
        self.snapshots_dir = os.path.join(self.script_dir, "snapshots")

    def _load(self, filename: str) -> Dict:
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                return {"error": str(e)}
        return {}

    def get_dashboard_export(self) -> Dict:
        """Get the combined dashboard export, or build it on-the-fly."""
        export = self._load("dashboard_export.json")
        if export and "account" in export:
            return export

        # Build on-the-fly from individual files
        return self._build_live_export()

    def _build_live_export(self) -> Dict:
        portfolio = self._load("portfolio_status.json")
        recommendations = self._load("recommendations.json")
        regime_ctx = self._load("regime_context.json")
        intel = self._load("market_intel.json")
        sentiment = self._load("news_sentiment.json")
        earnings = self._load("earnings_calendar.json")
        economic = self._load("economic_calendar.json")

        acct = portfolio.get("account", {})
        raw_pos = portfolio.get("positions", [])
        if isinstance(raw_pos, dict):
            raw_pos = list(raw_pos.values())

        positions = []
        for p in raw_pos:
            if isinstance(p, dict) and "symbol" in p:
                positions.append({
                    "symbol": p.get("symbol", ""),
                    "qty": float(p.get("qty", 0)),
                    "entry": float(p.get("avg_entry_price", 0)),
                    "current": float(p.get("current_price", 0)),
                    "pnl_pct": float(p.get("unrealized_plpc", 0)),
                    "sector": p.get("sector", "Unknown"),
                    "score": p.get("score", 50),
                    "protection": p.get("protection_type", "OCO"),
                    "alloc": float(p.get("allocation_pct", 0)),
                })

        signals = []
        for sig in recommendations.get("buy_signals", [])[:10]:
            signals.append({
                "symbol": sig.get("symbol", ""),
                "action": "BUY",
                "confidence": sig.get("confidence", 0),
                "agreement": sig.get("ai_agreement", ""),
                "sector": sig.get("sector", ""),
                "reasoning": sig.get("reasoning", ""),
            })
        for sig in recommendations.get("sell_signals", [])[:5]:
            signals.append({
                "symbol": sig.get("symbol", ""),
                "action": "SELL",
                "confidence": sig.get("confidence", 0),
                "agreement": sig.get("ai_agreement", ""),
                "sector": sig.get("sector", ""),
                "reasoning": sig.get("reasoning", ""),
            })

        news_items = []
        for item in (intel.get("items", []))[:20]:
            news_items.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "time": item.get("published", ""),
                "sentiment": item.get("sentiment", "neutral"),
                "category": item.get("category", ""),
            })

        return {
            "export_timestamp": datetime.now().isoformat(),
            "account": {
                "equity": acct.get("equity", 0),
                "cash": acct.get("cash", 0),
                "buying_power": acct.get("buying_power", 0),
                "daily_pnl": acct.get("daily_pnl", 0),
                "daily_pnl_pct": acct.get("daily_pnl_pct", 0),
                "positions_count": len(positions),
            },
            "positions": positions,
            "regime": {
                "name": regime_ctx.get("regime", "UNKNOWN").upper(),
                "score": regime_ctx.get("regime_score", 0),
                "risk_appetite": regime_ctx.get("risk_appetite", 0.5),
                "stop_loss": regime_ctx.get("recommended_stop_loss_pct", 7.0),
                "take_profit": regime_ctx.get("recommended_take_profit_pct", 20.0),
                "cash_reserve": regime_ctx.get("recommended_cash_reserve_pct", 3.0),
                "max_positions": regime_ctx.get("recommended_max_positions", 20),
            },
            "signals": signals,
            "news": news_items,
            "protection": portfolio.get("protection_summary", {}),
            "scoring": {"ai_confidence": 30, "momentum": 25, "technical": 20, "pnl": 15, "trend": 10},
            "sentiment_summary": sentiment.get("overall", {}),
            "earnings_upcoming": earnings.get("events", [])[:10],
            "economic_indicators": economic.get("indicators", []),
        }

    def get_positions(self) -> Dict:
        return self._load("portfolio_status.json")

    def get_signals(self) -> Dict:
        return self._load("recommendations.json")

    def get_news(self) -> Dict:
        return self._load("market_intel.json")

    def get_regime(self) -> Dict:
        return self._load("regime_context.json")

    def get_earnings(self) -> Dict:
        return self._load("earnings_calendar.json")

    def get_economic(self) -> Dict:
        return self._load("economic_calendar.json")

    def get_sentiment(self) -> Dict:
        return self._load("news_sentiment.json")

    def get_performance(self) -> Dict:
        return self._load("signal_performance.json")

    def list_snapshots(self):
        if not os.path.exists(self.snapshots_dir):
            return []
        return sorted([d for d in os.listdir(self.snapshots_dir)
                       if os.path.isdir(os.path.join(self.snapshots_dir, d))], reverse=True)

    def get_snapshot(self, date_str: str) -> Dict:
        snap_dir = os.path.join(self.snapshots_dir, date_str)
        if not os.path.exists(snap_dir):
            return {"error": f"Snapshot not found: {date_str}"}
        result = {}
        for fname in os.listdir(snap_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(snap_dir, fname), "r") as f:
                        result[fname.replace(".json", "")] = json.load(f)
                except Exception:
                    pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

store = DataStore()


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # API routes
        routes = {
            "/api/status": lambda: store.get_dashboard_export(),
            "/api/positions": lambda: store.get_positions(),
            "/api/signals": lambda: store.get_signals(),
            "/api/news": lambda: store.get_news(),
            "/api/regime": lambda: store.get_regime(),
            "/api/earnings": lambda: store.get_earnings(),
            "/api/economic": lambda: store.get_economic(),
            "/api/sentiment": lambda: store.get_sentiment(),
            "/api/performance": lambda: store.get_performance(),
            "/api/snapshots": lambda: {"snapshots": store.list_snapshots()},
        }

        if path in routes:
            self._json_response(routes[path]())
            return

        # Snapshot endpoint: /api/snapshot/2026-03-11
        if path.startswith("/api/snapshot/"):
            date_str = path.split("/")[-1]
            self._json_response(store.get_snapshot(date_str))
            return

        # Serve index.html for root
        if path == "" or path == "/":
            self._serve_index()
            return

        # 404
        self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data: Dict, status: int = 200):
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Luxverum Capital — Trading Terminal</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0A0E1A; color: #E2E8F0; font-family: 'Segoe UI', system-ui, sans-serif; }}
  .header {{ background: linear-gradient(135deg, #0A0E1A, #0D1321); border-bottom: 1px solid #1E2A42; padding: 20px 30px; display: flex; justify-content: space-between; align-items: center; }}
  .logo {{ display: flex; align-items: center; gap: 14px; }}
  .logo-icon {{ width: 40px; height: 40px; border-radius: 8px; background: linear-gradient(135deg, #D4A843, #8B7332); display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 20px; color: #0A0E1A; }}
  .logo-text {{ font-weight: 800; font-size: 18px; letter-spacing: 3px; color: #D4A843; font-family: monospace; }}
  .logo-sub {{ font-size: 11px; color: #64748B; letter-spacing: 1px; }}
  .content {{ max-width: 1200px; margin: 40px auto; padding: 0 30px; }}
  h2 {{ color: #D4A843; font-size: 16px; letter-spacing: 1px; margin-bottom: 16px; }}
  .endpoint {{ background: #151C2C; border: 1px solid #1E2A42; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
  .endpoint:hover {{ border-color: #D4A843; }}
  .method {{ background: #10B98118; color: #10B981; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }}
  .path {{ font-family: monospace; color: #E2E8F0; font-size: 13px; }}
  .desc {{ color: #64748B; font-size: 12px; }}
  a {{ color: #D4A843; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .note {{ background: #1A223540; border: 1px solid #1E2A42; border-radius: 8px; padding: 16px; margin-top: 20px; font-size: 12px; color: #64748B; line-height: 1.6; }}
  .note code {{ background: #0A0E1A; padding: 2px 6px; border-radius: 3px; color: #D4A843; font-size: 11px; }}
</style>
</head>
<body>
  <div class="header">
    <div class="logo">
      <div class="logo-icon">L</div>
      <div>
        <div class="logo-text">LUXVERUM CAPITAL</div>
        <div class="logo-sub">DASHBOARD API SERVER</div>
      </div>
    </div>
    <div style="font-size: 12px; color: #64748B;">
      Server started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
  </div>
  <div class="content">
    <h2>API ENDPOINTS</h2>
    <a href="/api/status" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/status</span></div><span class="desc">Full dashboard data export</span></a>
    <a href="/api/positions" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/positions</span></div><span class="desc">Current portfolio positions</span></a>
    <a href="/api/signals" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/signals</span></div><span class="desc">AI consensus signals</span></a>
    <a href="/api/news" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/news</span></div><span class="desc">Market intelligence feed</span></a>
    <a href="/api/regime" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/regime</span></div><span class="desc">Current market regime</span></a>
    <a href="/api/earnings" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/earnings</span></div><span class="desc">Upcoming earnings calendar</span></a>
    <a href="/api/economic" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/economic</span></div><span class="desc">FRED economic indicators</span></a>
    <a href="/api/sentiment" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/sentiment</span></div><span class="desc">News sentiment analysis</span></a>
    <a href="/api/performance" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/performance</span></div><span class="desc">Signal performance stats</span></a>
    <a href="/api/snapshots" class="endpoint"><div><span class="method">GET</span> <span class="path">/api/snapshots</span></div><span class="desc">Available snapshot dates</span></a>
    <div class="endpoint"><div><span class="method">GET</span> <span class="path">/api/snapshot/:date</span></div><span class="desc">Historical snapshot by date</span></div>

    <div class="note">
      <strong>Usage:</strong> Open the React dashboard and paste the data from <code>/api/status</code> into the import panel.<br>
      Or fetch from your dashboard: <code>fetch('http://localhost:8080/api/status').then(r => r.json())</code><br><br>
      <strong>News Collection:</strong> Run <code>python 11_news_aggregator.py --full</code> to populate the news feed,
      or start the daemon with <code>python 11_news_aggregator.py --daemon --interval 30</code>
    </div>
  </div>
</body>
</html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Custom log format."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {args[0]} {args[1]}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Luxverum Capital - Dashboard Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--refresh", action="store_true", help="Collect news before starting")
    args = parser.parse_args()

    if args.refresh:
        try:
            from importlib import import_module
            mod = import_module("11_news_aggregator")
            mod.run_full_collection()
        except Exception as e:
            print(f"  Warning: Could not refresh news: {e}")

    print_header("LUXVERUM CAPITAL — DASHBOARD SERVER")
    print(f"  Host:    {args.host}")
    print(f"  Port:    {args.port}")
    print(f"  URL:     http://{args.host}:{args.port}")
    print(f"  API:     http://{args.host}:{args.port}/api/status")
    print(f"  Data:    {store.data_dir}")
    print(f"\n  Press Ctrl+C to stop\n")

    server = HTTPServer((args.host, args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
