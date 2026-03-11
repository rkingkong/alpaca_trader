# ============================================================
# Luxverum Capital - Git Setup & Push Script
# ============================================================
# Run this from PowerShell in the "Trader Version 1" directory
#
# BEFORE RUNNING:
#   1. Make sure git is installed:  git --version
#   2. Make sure you're logged into GitHub:  gh auth status
#      (or have configured git credentials)
#   3. Place .gitignore, README.md, config.json.example,
#      and requirements.txt in "Trader Version 1/"
# ============================================================

# Step 0: Verify we're in the right directory
if (-not (Test-Path "run_trading_system.py")) {
    Write-Host "ERROR: Not in the Trader Version 1 directory!" -ForegroundColor Red
    Write-Host "Run:  cd 'C:\Users\kongr\OneDrive\ARMKU\Alpaca Trader\Trader Version 1'"
    exit 1
}

Write-Host "`n=== LUXVERUM CAPITAL - GIT SETUP ===" -ForegroundColor Cyan

# Step 1: Safety check - ensure .env and config.json won't be committed
Write-Host "`n[1/6] Verifying .gitignore exists..." -ForegroundColor Yellow
if (-not (Test-Path ".gitignore")) {
    Write-Host "  ERROR: .gitignore not found! Copy it here first." -ForegroundColor Red
    exit 1
}
Write-Host "  OK" -ForegroundColor Green

# Step 2: Initialize git (skip if already initialized)
Write-Host "`n[2/6] Initializing git..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init
    Write-Host "  Git initialized" -ForegroundColor Green
} else {
    Write-Host "  Git already initialized" -ForegroundColor Green
}

# Step 3: Set remote
Write-Host "`n[3/6] Setting remote origin..." -ForegroundColor Yellow
$remotes = git remote 2>&1
if ($remotes -match "origin") {
    git remote set-url origin https://github.com/rkingkong/alpaca_trader.git
    Write-Host "  Updated remote" -ForegroundColor Green
} else {
    git remote add origin https://github.com/rkingkong/alpaca_trader.git
    Write-Host "  Added remote" -ForegroundColor Green
}

# Step 4: Stage the right files
Write-Host "`n[4/6] Staging files..." -ForegroundColor Yellow
git add .gitignore
git add README.md
git add requirements.txt
git add config.json.example
git add config.py
git add ARCHITECTURE.md
git add run_trading_system.py
git add 00_regime_detector.py
git add 01_market_analysis.py
git add 02_portfolio_status.py
git add 03_execution_engine.py
git add 04_trade_sheet.py
git add 06_options_engine.py
git add 07_backtest_engine.py
git add 08_daily_snapshot.py
git add 09_live_monitor.py
git add verify_trailing_stops.py

# Show what's staged
Write-Host "`n  Staged files:" -ForegroundColor Cyan
git diff --cached --name-only

# Step 5: Verify nothing sensitive is staged
Write-Host "`n[5/6] Safety check - ensuring no secrets..." -ForegroundColor Yellow
$staged = git diff --cached --name-only
$dangerous = @(".env", "config.json")
foreach ($d in $dangerous) {
    if ($staged -match [regex]::Escape($d) -and $d -ne "config.json.example") {
        Write-Host "  DANGER: $d is staged! Aborting!" -ForegroundColor Red
        git reset HEAD $d
        exit 1
    }
}
Write-Host "  OK - no secrets in staging" -ForegroundColor Green

# Step 6: Commit and push
Write-Host "`n[6/6] Committing and pushing..." -ForegroundColor Yellow
git commit -m "v2.0: Consolidated architecture with dual-AI consensus engine

- 6-module architecture: regime, analysis, execution, options, backtest, snapshot
- Dual-AI consensus (Claude + GPT-4o) with conviction-based sizing
- Regime-adaptive parameters (bull/sideways/bear)
- Comprehensive protection: OCO brackets, trailing stops, graduation
- Protection quality audit: qty validation, staleness detection
- Complete rebalance dashboard with health checks
- Replaces old 5-file architecture with consolidated execution engine"

git branch -M main
git push -u origin main --force

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Repo: https://github.com/rkingkong/alpaca_trader" -ForegroundColor Cyan
Write-Host "`nVerify at the URL above that:"
Write-Host "  - .env is NOT listed"
Write-Host "  - config.json is NOT listed"
Write-Host "  - README.md shows correctly"
Write-Host "  - .gitignore is present"