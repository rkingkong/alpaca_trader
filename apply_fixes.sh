#!/usr/bin/env bash
# ================================================================
#  Luxverum Capital — System Patch  (2026-03-03 deep dive)
# ================================================================
#  Applies all 10 fixes identified in the code review.
#
#  USAGE:
#    1. Copy apply_fixes.sh + apply_fixes.py into "Trader Version 1/"
#    2. bash apply_fixes.sh
#    3. python run_trading_system.py --rebalance --dry-run
#
#  Backups are created automatically (*.bak)
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "Working directory: $SCRIPT_DIR"
echo ""

# Verify key files exist
for f in 01_market_analysis.py 03_execution_engine.py ARCHITECTURE.md; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found.  Run this from 'Trader Version 1/' folder."
        exit 1
    fi
done

python3 apply_fixes.py

echo "Done.  Test with:  python run_trading_system.py --rebalance --dry-run"