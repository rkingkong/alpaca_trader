#!/usr/bin/env python3
"""
Luxverum Capital -- Trading System Orchestrator
================================================
Single entry point for all trading operations.
Always use this instead of running modules directly.

Workflows:
  --rebalance    : Regime -> Analyze -> Status -> Full rebalance -> Protect -> Intelligence -> Options
  --full         : Regime -> Analyze -> Status -> New trades only -> Protect -> Intelligence -> Options
  --protect      : Regime -> Status -> Fix unprotected + Graduate winners
  --audit        : Status -> Read-only protection check
  --analyze      : Regime -> AI market analysis only
  --status       : Portfolio status only
  --snapshot     : Capture daily portfolio snapshot
  --options      : Run options strategies engine
  --backtest     : Run historical backtesting
  --intelligence : Run signal intelligence (log + stats + feedback + patterns + postmortem)
  --live         : Start 24/7 WebSocket monitor

Why orchestrator-first:
  Running modules directly skips the regime detector and uses default parameters
  instead of bull/bear-adjusted ones. The orchestrator ensures regime context
  is always fresh before any trading decisions.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_handler)


# ── Helpers ──────────────────────────────────────────────────────────────────

def path(filename: str) -> str:
    """Resolve a script path relative to SCRIPT_DIR."""
    return os.path.join(SCRIPT_DIR, filename)


def header(text: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def step(n: int, total: int, text: str):
    """Print a step divider."""
    print(f"\n{'-' * 70}")
    print(f"  Step {n}/{total}: {text}")
    print(f"{'-' * 70}\n")


def run(name: str, script: str, args: list = None, critical: bool = True) -> bool:
    """
    Run a module via subprocess.
    Returns True if successful, False otherwise.
    If critical=True, a failure is reported as a real failure.
    If critical=False, failure is non-fatal (returns True to not break chain).
    """
    script_path = path(script)
    if not os.path.exists(script_path):
        print(f"  [SKIP] Not found: {script_path}")
        return not critical  # Non-critical missing files are OK

    logger.info(f"Running: {name} ({sys.executable} {script_path})")

    try:
        cmd = [sys.executable, script_path] + (args or [])
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        logger.info(f"{name} completed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"{name} failed")
        return not critical
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user during {name}")
        return False
    except Exception as e:
        logger.error(f"{name} error: {e}")
        return not critical


def check_market_status() -> str:
    """Check if the market is currently open."""
    try:
        from alpaca.trading.client import TradingClient
        sys.path.insert(0, SCRIPT_DIR)
        from config import Config
        cfg = Config()
        client = TradingClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY,
            paper=cfg.PAPER_TRADING
        )
        clock = client.get_clock()
        status = "OPEN" if clock.is_open else "CLOSED"
        logger.info(f"Market status: {status}")
        return status
    except Exception:
        return "UNKNOWN"


def print_banner(workflow_name: str, dry_run: bool):
    """Print the startup banner."""
    market = check_market_status()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header("LUXVERUM CAPITAL -- TRADING SYSTEM")
    print(f"  Workflow:  {workflow_name}")
    print(f"  Mode:      {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"  Market:    {market}")
    print(f"  Time:      {now}")


def print_summary(results: dict, workflow_name: str, dry_run: bool):
    """Print execution summary."""
    if not results:
        return

    header("EXECUTION SUMMARY")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    market = check_market_status()

    print(f"  Workflow:  {workflow_name}")
    print(f"  Mode:      {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"  Time:      {now}")
    print(f"  Market:    {market}")

    all_ok = True
    for name, ok in results.items():
        tag = "[OK]" if ok else "[FAIL]"
        display = name.replace("_", " ").title()
        print(f"  {tag} {display}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  All steps completed successfully.")
    else:
        print(f"\n  Some steps had issues -- check output above.")


# ── Workflows ────────────────────────────────────────────────────────────────

def workflow_rebalance(dry_run: bool, force: bool) -> dict:
    """
    Full pipeline: Regime -> AI Analysis -> Status -> Full Rebalance
                   -> Protect -> Signal Intelligence -> Options Hedge
    """
    results = {}
    total = 7

    step(1, total, "Market Regime Detection")
    results["regime"] = run("Regime Detector", "00_regime_detector.py", critical=False)

    step(2, total, "AI Market Analysis")
    results["analysis"] = run("Market Analysis", "01_market_analysis.py")

    step(3, total, "Portfolio Status")
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")

    step(4, total, "Full Portfolio Rebalance")
    _args = ["--rebalance"]
    if dry_run: _args.append("--dry-run")
    if force: _args.append("--force")
    results["rebalance"] = run("Execution Engine", "03_execution_engine.py", _args)

    step(5, total, "Protection Verification")
    _prot = ["--protect-only"]
    if dry_run: _prot.append("--dry-run")
    results["protection"] = run("Protection Verification", "03_execution_engine.py", _prot, critical=False)

    step(6, total, "Signal Intelligence")
    _intel = ["--log", "--stats", "--feedback", "--patterns"]
    results["intelligence"] = run("Signal Intelligence", "09_signal_intelligence.py", _intel, critical=False)

    step(7, total, "Institutional Options Overlay")
    _opt = ["--strategy", "collar", "macro"]
    if dry_run: _opt.append("--dry-run")
    if force: _opt.append("--force")
    results["options_hedge"] = run("Options Engine", "06_options_engine.py", _opt, critical=False)

    return results


def workflow_full(dry_run: bool, force: bool) -> dict:
    """
    New trades only: Regime -> AI Analysis -> Status -> New Trades
                     -> Protect -> Signal Intelligence -> Options Hedge
    """
    results = {}
    total = 7

    step(1, total, "Market Regime Detection")
    results["regime"] = run("Regime Detector", "00_regime_detector.py", critical=False)

    step(2, total, "AI Market Analysis")
    results["analysis"] = run("Market Analysis", "01_market_analysis.py")

    step(3, total, "Portfolio Status")
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")

    step(4, total, "New Trade Execution")
    _args = ["--new-trades"]
    if dry_run: _args.append("--dry-run")
    if force: _args.append("--force")
    results["new_trades"] = run("Execution Engine", "03_execution_engine.py", _args)

    step(5, total, "Protection Verification")
    _prot = ["--protect-only"]
    if dry_run: _prot.append("--dry-run")
    results["protection"] = run("Protection Verification", "03_execution_engine.py", _prot, critical=False)

    step(6, total, "Signal Intelligence")
    _intel = ["--log", "--stats", "--feedback", "--patterns"]
    results["intelligence"] = run("Signal Intelligence", "09_signal_intelligence.py", _intel, critical=False)

    step(7, total, "Institutional Options Overlay")
    _opt = ["--strategy", "collar", "macro"]
    if dry_run: _opt.append("--dry-run")
    if force: _opt.append("--force")
    results["options_hedge"] = run("Options Engine", "06_options_engine.py", _opt, critical=False)

    return results


def workflow_protect(dry_run: bool) -> dict:
    """
    Quick protection fix: Regime -> Status -> Protect + Graduate
    Runs regime first so trail percentages are correct.
    """
    results = {}
    total = 3

    step(1, total, "Market Regime Detection")
    results["regime"] = run("Regime Detector", "00_regime_detector.py", critical=False)

    step(2, total, "Portfolio Status")
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")

    step(3, total, "Fix Protection + Graduate Winners")
    _args = ["--protect-only"]
    if dry_run: _args.append("--dry-run")
    results["protection"] = run("Protection Fix", "03_execution_engine.py", _args)

    return results


def workflow_audit() -> dict:
    """Read-only protection check. No changes, no trades."""
    results = {}
    total = 2

    step(1, total, "Portfolio Status")
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")

    step(2, total, "Protection Audit (Read-Only)")
    results["audit"] = run("Protection Audit", "03_execution_engine.py", ["--audit"])

    return results


def workflow_analyze() -> dict:
    """Run AI market analysis only (Regime + Claude + GPT-4o)."""
    results = {}
    total = 2

    step(1, total, "Market Regime Detection")
    results["regime"] = run("Regime Detector", "00_regime_detector.py", critical=False)

    step(2, total, "AI Market Analysis")
    results["analysis"] = run("Market Analysis", "01_market_analysis.py")

    return results


def workflow_status() -> dict:
    """Show current portfolio status."""
    results = {}
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")
    return results


def workflow_snapshot() -> dict:
    """Capture daily portfolio snapshot + reconciliation."""
    results = {}
    results["snapshot"] = run("Daily Snapshot", "08_daily_snapshot.py")
    return results


def workflow_options(strategy_args: list, dry_run: bool, force: bool) -> dict:
    """Run options strategies engine."""
    results = {}
    total = 2

    step(1, total, "Portfolio Status")
    results["status"] = run("Portfolio Status", "02_portfolio_status.py")

    step(2, total, "Options Strategy Engine")
    _args = []
    if strategy_args:
        _args.extend(["--strategy"] + strategy_args)
    if dry_run: _args.append("--dry-run")
    if force: _args.append("--force")
    results["options"] = run("Options Engine", "06_options_engine.py", _args)

    return results


def workflow_backtest() -> dict:
    """Run historical backtesting engine."""
    results = {}
    results["backtest"] = run("Backtest Engine", "07_backtest_engine.py")
    return results


def workflow_intelligence() -> dict:
    """
    Run full signal intelligence: log + stats + feedback + patterns + postmortem.
    """
    results = {}
    results["intelligence"] = run(
        "Signal Intelligence (Full)",
        "09_signal_intelligence.py",
        ["--full"],
    )
    return results


def workflow_live() -> dict:
    """Start the 24/7 WebSocket monitor."""
    header("STARTING EVENT-DRIVEN WEBSOCKET MONITOR")
    print("  Connecting to Alpaca live stream. Press Ctrl+C to stop.")
    return {"live_monitor": run("Live Monitor", "10_live_monitor.py")}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Luxverum Capital -- Trading System Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --rebalance --dry-run     Preview full rebalance pipeline
  %(prog)s --rebalance               Execute live rebalance
  %(prog)s --protect                 Fix unprotected positions + graduate
  %(prog)s --audit                   Read-only protection check
  %(prog)s --analyze                 Run AI analysis only
  %(prog)s --intelligence            Run signal intelligence engine
  %(prog)s --options --strategy collar macro
"""
    )

    # Workflow selection (mutually exclusive)
    wf = parser.add_mutually_exclusive_group()
    wf.add_argument("--rebalance", action="store_true",
                     help="Full pipeline: Regime -> AI -> Status -> Rebalance -> Protect -> Intelligence -> Options")
    wf.add_argument("--full", action="store_true",
                     help="New trades only: Regime -> AI -> Status -> New trades -> Protect -> Intelligence -> Options")
    wf.add_argument("--protect", action="store_true",
                     help="Regime -> Status -> Fix unprotected + Graduate winners")
    wf.add_argument("--audit", action="store_true",
                     help="Read-only protection check (no changes)")
    wf.add_argument("--analyze", action="store_true",
                     help="Only run AI market analysis (Claude + GPT-4o)")
    wf.add_argument("--status", action="store_true",
                     help="Show current portfolio status")
    wf.add_argument("--snapshot", action="store_true",
                     help="Capture daily portfolio snapshot")
    wf.add_argument("--options", action="store_true",
                     help="Run options strategies engine")
    wf.add_argument("--backtest", action="store_true",
                     help="Run historical backtesting")
    wf.add_argument("--intelligence", action="store_true",
                     help="Run signal intelligence (log + stats + feedback + patterns + postmortem)")
    wf.add_argument("--live", action="store_true",
                     help="Start 24/7 WebSocket monitor")

    # Modifiers
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview mode -- no live trades")
    parser.add_argument("--force", action="store_true",
                        help="Force execution even if safety checks fail")
    parser.add_argument("--strategy", nargs="*", default=[],
                        help="Options strategy names (e.g., collar macro spreads protect)")

    args = parser.parse_args()

    # ── Route to workflow ──

    if args.rebalance:
        print_banner("Full Portfolio Rebalancing + Hedging", args.dry_run)
        res = workflow_rebalance(args.dry_run, args.force)
        print_summary(res, "Rebalance", args.dry_run)

    elif args.full:
        print_banner("New Trades + Protection + Hedging", args.dry_run)
        res = workflow_full(args.dry_run, args.force)
        print_summary(res, "Full", args.dry_run)

    elif args.protect:
        print_banner("Protection Fix + Graduation", args.dry_run)
        res = workflow_protect(args.dry_run)
        print_summary(res, "Protect", args.dry_run)

    elif args.audit:
        print_banner("Protection Audit (Read-Only)", False)
        res = workflow_audit()
        print_summary(res, "Audit", False)

    elif args.analyze:
        print_banner("AI Market Analysis", False)
        res = workflow_analyze()
        print_summary(res, "Analyze", False)

    elif args.status:
        res = workflow_status()

    elif args.snapshot:
        res = workflow_snapshot()

    elif args.options:
        print_banner("Options Strategy Engine", args.dry_run)
        res = workflow_options(args.strategy, args.dry_run, args.force)
        print_summary(res, "Options", args.dry_run)

    elif args.backtest:
        header("LUXVERUM CAPITAL -- BACKTESTING ENGINE")
        res = workflow_backtest()

    elif args.intelligence:
        header("LUXVERUM CAPITAL -- SIGNAL INTELLIGENCE")
        res = workflow_intelligence()
        print_summary(res, "Intelligence", False)

    elif args.live:
        workflow_live()

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

# ═══════════════════════════════════════════════════════════════════
# TERMINAL UPGRADE v2.0 — Added 2026-03-11
# ═══════════════════════════════════════════════════════════════════
# To integrate news collection into your workflow, add these to
# workflow_rebalance() after the signal intelligence step:
#
#   step(N, total, "News & Market Intelligence")
#   results["news_intel"] = run("News Aggregator", "11_news_aggregator.py", ["--full"], critical=False)
#
#   step(N+1, total, "Dashboard Data Export")
#   results["dashboard_export"] = run("Dashboard Export", "11_news_aggregator.py", ["--export"], critical=False)
#
# Also increment 'total' by 2 at the top of workflow_rebalance().
#
# Or run standalone:
#   python 11_news_aggregator.py --full
#   python 10_dashboard_server.py
# ═══════════════════════════════════════════════════════════════════
