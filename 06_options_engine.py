#!/usr/bin/env python3
"""
Module 6: Options Strategy Engine (v4 — Empire Architecture)
========================================================================
Institutional options overlay providing 6 strategies:
  1. Bull Call Spreads     - Defined-risk leveraged upside
  2. PMCC                  - Capital-efficient LEAPS growth
  3. Cash-Secured Puts     - Income generation + discounted entries
  4. Protective Puts       - Standard downside protection
  5. Zero-Cost Collars     - FREE downside protection funded by selling calls
  6. Macro Hedge (SPY)     - Buying SPY Puts to protect the entire fund
"""

import os
import sys
import json
import time
import math
import logging
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
    OptionLegRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    AssetStatus,
    OrderSide,
    OrderClass,
    TimeInForce,
    ContractType,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockBarsRequest,
    OptionLatestQuoteRequest,
    OptionSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("OptionsEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)


class OptionsStrategyEngine:
    """
    Multi-strategy options engine for aggressive growth with structural protection.
    
    Reads AI recommendations from Module 1 and portfolio state from Module 2,
    then deploys options strategies through Alpaca's multi-leg order API.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION — Tune these to your risk appetite
    # ══════════════════════════════════════════════════════════════════════════

    # ── Capital Allocation ────────────────────────────────────────────────────
    MAX_OPTIONS_ALLOCATION_PCT = 30.0     # Max 30% of portfolio in options
    MAX_SINGLE_TRADE_PCT = 3.0            # Max 3% risk on any single options trade
    MIN_BUYING_POWER_RESERVE_PCT = 15.0   # Always keep 15% buying power free
    CSP_CASH_RESERVE_PCT = 50.0           # Max 50% of cash reserved for CSPs

    # ── Bull Call Spread Settings ─────────────────────────────────────────────
    BCS_MIN_CONFIDENCE = 0.70             # Min AI confidence for bull call spreads
    BCS_EXPIRATION_MIN_DAYS = 21          # Min 21 DTE for spreads
    BCS_EXPIRATION_MAX_DAYS = 60          # Max 60 DTE for spreads
    BCS_STRIKE_WIDTH_PCT = 10.0           # Spread width as % of underlying price
    BCS_TARGET_DELTA_LONG = (0.45, 0.65)  # ATM-ish long call delta range
    BCS_TARGET_DELTA_SHORT = (0.20, 0.40) # OTM short call delta range
    BCS_MIN_OPEN_INTEREST = 50            # Min OI for liquidity
    BCS_TARGET_PROFIT_PCT = 50.0          # Close at 50% of max profit
    BCS_MAX_LOSS_PCT = 100.0              # Let spread expire worthless (defined risk)

    # ── PMCC (Poor Man's Covered Call) Settings ───────────────────────────────
    PMCC_MIN_CONFIDENCE = 0.75            # Higher bar — this is a longer commitment
    PMCC_LEAPS_MIN_DAYS = 180             # Min 6 months for LEAPS leg
    PMCC_LEAPS_MAX_DAYS = 730             # Max 2 years for LEAPS leg
    PMCC_LEAPS_TARGET_DELTA = (0.70, 0.85)  # Deep ITM LEAPS
    PMCC_SHORT_MIN_DAYS = 14              # Min 2 weeks for short call
    PMCC_SHORT_MAX_DAYS = 45              # Max 45 DTE for short call
    PMCC_SHORT_TARGET_DELTA = (0.20, 0.35)  # OTM short call
    PMCC_MIN_OPEN_INTEREST = 100          # Higher OI needed for LEAPS

    # ── Cash-Secured Put Settings ─────────────────────────────────────────────
    CSP_MIN_CONFIDENCE = 0.60             # Moderate confidence — we want to own it
    CSP_EXPIRATION_MIN_DAYS = 14          # Min 2 weeks
    CSP_EXPIRATION_MAX_DAYS = 45          # Max 45 DTE — sweet spot for theta
    CSP_TARGET_DELTA = (0.20, 0.35)       # OTM puts — 20-35 delta
    CSP_DISCOUNT_TARGET_PCT = 5.0         # Target 5% below current price
    CSP_MIN_OPEN_INTEREST = 50
    CSP_MIN_PREMIUM_PCT = 1.0             # Min 1% premium yield

    # ── Protective Put Settings ───────────────────────────────────────────────
    PROTECT_MIN_POSITION_VALUE = 500.0    # Only protect positions worth $500+
    PROTECT_EXPIRATION_MIN_DAYS = 30      # Min 30 DTE for protection
    PROTECT_EXPIRATION_MAX_DAYS = 90      # Max 90 DTE for protection
    PROTECT_STRIKE_PCT_BELOW = 10.0       # Put strike 10% below current price
    PROTECT_MAX_COST_PCT = 2.0            # Max 2% of position value for protection
    
    # ── Zero-Cost Collar Settings ─────────────────────────────────────────────
    COLLAR_CALL_STRIKE_PCT_ABOVE = 10.0   # Target short call 10% above current price

    # ── Macro Hedge Settings ──────────────────────────────────────────────────
    MACRO_HEDGE_ALLOCATION_PCT = 1.0      # Use 1% of portfolio equity to hedge
    MACRO_HEDGE_TICKER = "SPY"
    MACRO_HEDGE_DTE = 45                  # Target 45 days out

    # ── Greeks & Pricing ──────────────────────────────────────────────────────
    RISK_FREE_RATE = 0.045                # ~4.5% risk-free rate (update periodically)
    IV_MAX_THRESHOLD = 0.80               # Skip if IV > 80% (too expensive)
    IV_MIN_THRESHOLD = 0.10               # Skip if IV < 10% (something's wrong)

    # ── Order Settings ────────────────────────────────────────────────────────
    ORDER_FILL_TIMEOUT_SECONDS = 60
    ORDER_POLL_INTERVAL_SECONDS = 2

    def __init__(self, config_path: str = None, dry_run: bool = False):
        """Initialize the options engine."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dry_run = dry_run
        
        self.vr_cache = {} # Cache para almacenar los rangos de volatilidad procesados

        if config_path is None:
            config_path = os.path.join(self.script_dir, "config.json")

        self._load_config(config_path)
        self._setup_clients()

        self.data_dir = os.path.join(self.script_dir, "data")
        self.logs_dir = os.path.join(self.script_dir, "logs")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Execution tracking
        self.execution_log = {
            "timestamp": datetime.now().isoformat(),
            "mode": "DRY RUN" if dry_run else "LIVE",
            "strategies_run": [],
            "trades_executed": [],
            "trades_skipped": [],
            "errors": [],
            "warnings": [],
            "summary": {}
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _load_config(self, config_path: str):
        """Load API credentials from config file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.alpaca_key = config.get('ALPACA_API_KEY', os.getenv('ALPACA_API_KEY'))
            self.alpaca_secret = config.get('ALPACA_SECRET_KEY', os.getenv('ALPACA_SECRET_KEY'))
            self.paper_trading = config.get('PAPER_TRADING', True)
        else:
            self.alpaca_key = os.getenv('ALPACA_API_KEY')
            self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
            self.paper_trading = True

    def _setup_clients(self):
        """Initialize Alpaca API clients."""
        self.trading_client = TradingClient(
            self.alpaca_key,
            self.alpaca_secret,
            paper=self.paper_trading
        )
        self.stock_data_client = StockHistoricalDataClient(
            self.alpaca_key,
            self.alpaca_secret
        )
        self.option_data_client = OptionHistoricalDataClient(
            self.alpaca_key,
            self.alpaca_secret
        )

    def _log_action(self, action: str, details: Dict):
        self.execution_log["trades_executed"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            **details
        })

    def _log_skip(self, reason: str, details: Dict = None):
        self.execution_log["trades_skipped"].append({
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            **(details or {})
        })

    def _log_error(self, error: str, context: Dict = None):
        self.execution_log["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {}
        })

    def _log_warning(self, warning: str, context: Dict = None):
        self.execution_log["warnings"].append({
            "timestamp": datetime.now().isoformat(),
            "warning": warning,
            "context": context or {}
        })

    @staticmethod
    def _is_option_symbol(symbol: str) -> bool:
        """
        Detect if a symbol is an option contract (OCC format).
        e.g., AAPL250620C00200000, SPY260116P00400000
        """
        import re
        return bool(symbol and len(symbol) >= 10 and re.match(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$', symbol))

    # ══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL EDGE: VOLATILITY RANK
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_volatility_rank(self, symbol: str) -> float:
        """
        Calcula el IV Rank (Proxy de Volatilidad Histórica de 1 año).
        Compara la volatilidad de los últimos 20 días contra la del año entero.
        0 a 100. Valores altos (>60) = Opciones caras. Valores bajos (<60) = Opciones baratas.
        """
        if symbol in self.vr_cache:
            return self.vr_cache[symbol]

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = self.stock_data_client.get_stock_bars(req)
            if not hasattr(bars, 'df') or bars.df.empty: 
                return 50.0
                
            df = bars.df.reset_index()
            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol]
                
            close = df['close']
            if len(close) < 30: 
                return 50.0
                
            returns = close.pct_change().dropna()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
            
            if rolling_vol.empty: 
                return 50.0
                
            current_vol = rolling_vol.iloc[-1]
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()
            
            if max_vol == min_vol: 
                return 50.0
                
            iv_rank = ((current_vol - min_vol) / (max_vol - min_vol)) * 100
            iv_rank = round(iv_rank, 1)
            
            self.vr_cache[symbol] = iv_rank
            return iv_rank
            
        except Exception as e:
            self._log_warning(f"Error calculando Vol Rank para {symbol}: {e}")
            return 50.0

    # ══════════════════════════════════════════════════════════════════════════
    # DATA LOADING — from Module 1 & 2 outputs
    # ══════════════════════════════════════════════════════════════════════════

    def load_recommendations(self) -> Optional[Dict]:
        """Load AI recommendations from Module 1."""
        filepath = os.path.join(self.data_dir, "recommendations.json")
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._log_error(f"Failed to load recommendations: {e}")
            return None

    def load_portfolio_status(self) -> Optional[Dict]:
        """Load portfolio status from Module 2."""
        filepath = os.path.join(self.data_dir, "portfolio_status.json")
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._log_error(f"Failed to load portfolio status: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE DATA — Account, Prices, Options Chain
    # ══════════════════════════════════════════════════════════════════════════

    def get_account(self) -> Dict:
        """Fetch current account info."""
        try:
            account = self.trading_client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "options_trading_level": getattr(account, 'options_trading_level', None),
                "status": account.status.value if hasattr(account.status, 'value') else str(account.status),
            }
        except Exception as e:
            self._log_error(f"Failed to fetch account: {e}")
            return {}

    def get_positions(self) -> Dict[str, Dict]:
        """Fetch all current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                    "asset_class": pos.asset_class.value if hasattr(pos.asset_class, 'value') else str(pos.asset_class)
                }
            return result
        except Exception as e:
            self._log_error(f"Failed to fetch positions: {e}")
            return {}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current mid-price for a stock symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data_client.get_stock_latest_quote(request)
            if symbol in quotes:
                quote = quotes[symbol]
                bid = float(quote.bid_price) if quote.bid_price else 0
                ask = float(quote.ask_price) if quote.ask_price else 0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return bid or ask or None
            return None
        except Exception as e:
            self._log_warning(f"Failed to get price for {symbol}: {e}")
            return None

    def get_option_chain(self, underlying_symbol: str, contract_type: ContractType,
                         min_expiration: date, max_expiration: date,
                         min_strike: float = None, max_strike: float = None,
                         limit: int = 100) -> List:
        """
        Fetch option contracts from Alpaca matching the given criteria.
        Returns list of OptionContract objects.
        """
        try:
            params = {
                "underlying_symbols": [underlying_symbol],
                "status": AssetStatus.ACTIVE,
                "type": contract_type,
                "expiration_date_gte": str(min_expiration),
                "expiration_date_lte": str(max_expiration),
                "limit": limit,
            }
            if min_strike is not None:
                params["strike_price_gte"] = str(min_strike)
            if max_strike is not None:
                params["strike_price_lte"] = str(max_strike)

            req = GetOptionContractsRequest(**params)
            response = self.trading_client.get_option_contracts(req)
            return response.option_contracts if response.option_contracts else []
        except Exception as e:
            self._log_error(f"Failed to fetch option chain for {underlying_symbol}: {e}")
            return []

    def get_option_quote(self, option_symbol: str) -> Optional[Dict]:
        """Get latest quote for an option contract."""
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol)
            quotes = self.option_data_client.get_option_latest_quote(req)
            if option_symbol in quotes:
                q = quotes[option_symbol]
                bid = float(q.bid_price) if q.bid_price else 0
                ask = float(q.ask_price) if q.ask_price else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
                return {
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "bid_size": int(q.bid_size) if q.bid_size else 0,
                    "ask_size": int(q.ask_size) if q.ask_size else 0,
                }
            return None
        except Exception as e:
            self._log_warning(f"Failed to get option quote for {option_symbol}: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # GREEKS CALCULATION (Black-Scholes)
    # ══════════════════════════════════════════════════════════════════════════

    def calculate_implied_volatility(self, option_price: float, S: float,
                                      K: float, T: float, r: float,
                                      option_type: str = "call") -> Optional[float]:
        """Calculate implied volatility using Brent's method."""
        if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
            return None
        try:
            def objective(sigma):
                return self._bs_price(S, K, T, r, sigma, option_type) - option_price

            iv = brentq(objective, 0.001, 5.0, maxiter=200)
            return iv
        except (ValueError, RuntimeError):
            return None

    def _bs_price(self, S: float, K: float, T: float, r: float,
                  sigma: float, option_type: str = "call") -> float:
        """Black-Scholes option price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calculate_greeks(self, S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str = "call") -> Optional[Dict]:
        """Calculate option Greeks (delta, gamma, theta, vega)."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == "call":
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                         - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                delta = -norm.cdf(-d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                         + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # per 1% IV move

            return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
        except Exception:
            return None

    def enrich_option(self, option_contract, underlying_price: float) -> Optional[Dict]:
        """
        Enrich a raw Alpaca OptionContract with live quote data and Greeks.
        Returns a dict with all data needed for strategy selection.
        """
        symbol = option_contract.symbol
        strike = float(option_contract.strike_price)
        exp_date = option_contract.expiration_date
        oi = int(option_contract.open_interest) if option_contract.open_interest else 0
        contract_type = "call" if option_contract.type == ContractType.CALL else "put"

        # Days to expiration
        if isinstance(exp_date, str):
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
        dte = (exp_date - date.today()).days
        T = dte / 365.0

        if T <= 0:
            return None

        # Get live quote
        quote = self.get_option_quote(symbol)
        if not quote or quote["mid"] <= 0:
            return None

        mid_price = quote["mid"]

        # Calculate IV and Greeks
        iv = self.calculate_implied_volatility(mid_price, underlying_price, strike, T,
                                                self.RISK_FREE_RATE, contract_type)
        greeks = None
        if iv and iv > 0:
            greeks = self.calculate_greeks(underlying_price, strike, T,
                                           self.RISK_FREE_RATE, iv, contract_type)

        return {
            "symbol": symbol,
            "underlying": option_contract.underlying_symbol,
            "strike": strike,
            "expiration_date": str(exp_date),
            "dte": dte,
            "type": contract_type,
            "open_interest": oi,
            "bid": quote["bid"],
            "ask": quote["ask"],
            "mid": mid_price,
            "iv": iv,
            "delta": greeks["delta"] if greeks else None,
            "gamma": greeks["gamma"] if greeks else None,
            "theta": greeks["theta"] if greeks else None,
            "vega": greeks["vega"] if greeks else None,
            "contract_size": 100,  # Standard option contract
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PREFLIGHT CHECKS
    # ══════════════════════════════════════════════════════════════════════════

    def preflight_check(self) -> Tuple[bool, Dict]:
        """
        Verify the account is ready for options trading.
        Returns (ok, account_info).
        """
        account = self.get_account()
        if not account:
            logger.error("❌ Cannot reach Alpaca API")
            return False, {}

        # Check options level
        options_level = account.get("options_trading_level")
        if options_level is not None and options_level < 2:
            logger.error(f"❌ Options level {options_level} — need at least Level 2 for spreads")
            return False, account

        # Check buying power
        equity = account.get("equity", 0)
        buying_power = account.get("buying_power", 0)
        min_bp = equity * (self.MIN_BUYING_POWER_RESERVE_PCT / 100)

        if buying_power < min_bp:
            logger.warning(f"⚠️  Low buying power: ${buying_power:,.2f} (min reserve: ${min_bp:,.2f})")
            self._log_warning("Low buying power", {"buying_power": buying_power, "min": min_bp})

        logger.info(f"✅ Account OK — Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f} | Options Level: {options_level}")
        return True, account

    def calculate_max_options_budget(self, account: Dict) -> float:
        """How much capital we can allocate to options trades right now."""
        equity = account.get("equity", 0)
        buying_power = account.get("buying_power", 0)

        # Max allocation from equity perspective
        max_from_equity = equity * (self.MAX_OPTIONS_ALLOCATION_PCT / 100)

        # Can't exceed actual buying power minus reserve
        reserve = equity * (self.MIN_BUYING_POWER_RESERVE_PCT / 100)
        available_bp = max(0, buying_power - reserve)

        budget = min(max_from_equity, available_bp)
        logger.info(f"  💰 Options budget: ${budget:,.2f} (max alloc: ${max_from_equity:,.2f}, available BP: ${available_bp:,.2f})")
        return budget

    def max_trade_size(self, account: Dict) -> float:
        """Max capital for any single options trade."""
        equity = account.get("equity", 0)
        return equity * (self.MAX_SINGLE_TRADE_PCT / 100)

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: BULL CALL SPREADS
    # ══════════════════════════════════════════════════════════════════════════

    def find_bull_call_spread(self, symbol: str, underlying_price: float,
                              max_cost: float) -> Optional[Dict]:
        """
        Find the optimal bull call spread for a symbol.
        """
        logger.info(f"  🔍 Scanning bull call spreads for {symbol} @ ${underlying_price:.2f}...")

        today = date.today()
        min_exp = today + timedelta(days=self.BCS_EXPIRATION_MIN_DAYS)
        max_exp = today + timedelta(days=self.BCS_EXPIRATION_MAX_DAYS)

        # Strike range around current price
        spread_width = underlying_price * (self.BCS_STRIKE_WIDTH_PCT / 100)
        min_strike = round(underlying_price * 0.95, 2)  # Slightly ITM for long
        max_strike = round(underlying_price * 1.15, 2)   # OTM for short

        # Fetch call options
        calls = self.get_option_chain(symbol, ContractType.CALL, min_exp, max_exp,
                                       min_strike, max_strike)
        if not calls:
            self._log_skip(f"No call options found for {symbol}", {"symbol": symbol})
            return None

        # Enrich with quotes and Greeks
        enriched = []
        for c in calls:
            data = self.enrich_option(c, underlying_price)
            if data and data["open_interest"] >= self.BCS_MIN_OPEN_INTEREST:
                if data["iv"] and self.IV_MIN_THRESHOLD < data["iv"] < self.IV_MAX_THRESHOLD:
                    enriched.append(data)

        if len(enriched) < 2:
            self._log_skip(f"Not enough liquid options for {symbol}", {"symbol": symbol, "found": len(enriched)})
            return None

        # Group by expiration
        by_expiration = {}
        for opt in enriched:
            exp = opt["expiration_date"]
            if exp not in by_expiration:
                by_expiration[exp] = []
            by_expiration[exp].append(opt)

        # Find best spread per expiration
        best_spread = None
        best_score = -1

        for exp_date, options in sorted(by_expiration.items()):
            options.sort(key=lambda x: x["strike"])

            # Find long call candidates (higher delta = more ITM)
            long_candidates = [o for o in options
                               if o["delta"] is not None
                               and self.BCS_TARGET_DELTA_LONG[0] <= abs(o["delta"]) <= self.BCS_TARGET_DELTA_LONG[1]]

            # Find short call candidates (lower delta = more OTM)
            short_candidates = [o for o in options
                                if o["delta"] is not None
                                and self.BCS_TARGET_DELTA_SHORT[0] <= abs(o["delta"]) <= self.BCS_TARGET_DELTA_SHORT[1]]

            for long_call in long_candidates:
                for short_call in short_candidates:
                    # Short must be higher strike than long
                    if short_call["strike"] <= long_call["strike"]:
                        continue

                    # Calculate spread economics
                    net_debit = (long_call["mid"] - short_call["mid"]) * 100
                    spread_width_dollars = (short_call["strike"] - long_call["strike"]) * 100
                    max_profit = spread_width_dollars - net_debit
                    max_loss = net_debit

                    # Skip if too expensive
                    if net_debit <= 0 or net_debit > max_cost:
                        continue

                    # Risk/reward ratio
                    if max_loss <= 0:
                        continue
                    rr_ratio = max_profit / max_loss

                    # Score: higher R:R + tighter spread + mid-range DTE
                    dte = long_call["dte"]
                    dte_score = 1 - abs(dte - 35) / 35  # Prefer ~35 DTE
                    score = (rr_ratio * 0.5) + (dte_score * 0.3) + (min(rr_ratio, 3) / 3 * 0.2)

                    if score > best_score:
                        best_score = score
                        best_spread = {
                            "strategy": "bull_call_spread",
                            "symbol": symbol,
                            "underlying_price": underlying_price,
                            "long_call": long_call,
                            "short_call": short_call,
                            "net_debit": round(net_debit, 2),
                            "max_profit": round(max_profit, 2),
                            "max_loss": round(max_loss, 2),
                            "risk_reward": round(rr_ratio, 2),
                            "breakeven": round(long_call["strike"] + net_debit / 100, 2),
                            "expiration": exp_date,
                            "dte": long_call["dte"],
                            "score": round(score, 3),
                        }

        if best_spread:
            logger.info(f"    ✅ Found BCS: Buy ${best_spread['long_call']['strike']}C / "
                        f"Sell ${best_spread['short_call']['strike']}C | "
                        f"Cost: ${best_spread['net_debit']:.2f} | "
                        f"Max Profit: ${best_spread['max_profit']:.2f} | "
                        f"R:R {best_spread['risk_reward']:.1f}x")
        else:
            self._log_skip(f"No profitable BCS found for {symbol}", {"symbol": symbol})

        return best_spread

    def execute_bull_call_spread(self, spread: Dict) -> Dict:
        """Execute a bull call spread via multi-leg order."""
        long_sym = spread["long_call"]["symbol"]
        short_sym = spread["short_call"]["symbol"]

        logger.info(f"  📥 Executing BCS: Buy {long_sym} / Sell {short_sym}")

        if self.dry_run:
            result = {"success": True, "dry_run": True, **spread}
            self._log_action("bull_call_spread", result)
            return result

        try:
            order_legs = [
                OptionLegRequest(
                    symbol=long_sym,
                    side=OrderSide.BUY,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=short_sym,
                    side=OrderSide.SELL,
                    ratio_qty=1
                )
            ]

            req = MarketOrderRequest(
                qty=1,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=order_legs
            )
            res = self.trading_client.submit_order(req)
            order_id = str(res.id)

            result = {
                "success": True,
                "order_id": order_id,
                "status": res.status.value if hasattr(res.status, 'value') else str(res.status),
                **spread
            }
            self._log_action("bull_call_spread", result)
            logger.info(f"    ✅ BCS order submitted: {order_id}")
            return result

        except Exception as e:
            self._log_error(f"BCS execution failed for {spread['symbol']}: {e}")
            return {"success": False, "error": str(e), **spread}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: POOR MAN'S COVERED CALL (PMCC)
    # ══════════════════════════════════════════════════════════════════════════

    def find_pmcc(self, symbol: str, underlying_price: float,
                  max_cost: float) -> Optional[Dict]:
        """
        Find an optimal PMCC (diagonal call spread).
        """
        logger.info(f"  🔍 Scanning PMCC setups for {symbol} @ ${underlying_price:.2f}...")

        today = date.today()

        # === LEAPS Leg (deep ITM, long-dated) ===
        leaps_min_exp = today + timedelta(days=self.PMCC_LEAPS_MIN_DAYS)
        leaps_max_exp = today + timedelta(days=self.PMCC_LEAPS_MAX_DAYS)
        leaps_max_strike = round(underlying_price * 0.85, 2)  # At least 15% ITM

        leaps_calls = self.get_option_chain(
            symbol, ContractType.CALL, leaps_min_exp, leaps_max_exp,
            max_strike=leaps_max_strike, limit=50
        )

        if not leaps_calls:
            self._log_skip(f"No LEAPS found for {symbol}", {"symbol": symbol})
            return None

        # Enrich LEAPS candidates
        leaps_enriched = []
        for c in leaps_calls:
            data = self.enrich_option(c, underlying_price)
            if (data and data["open_interest"] >= self.PMCC_MIN_OPEN_INTEREST
                    and data["delta"] is not None
                    and self.PMCC_LEAPS_TARGET_DELTA[0] <= abs(data["delta"]) <= self.PMCC_LEAPS_TARGET_DELTA[1]):
                # Cost must be within budget
                cost = data["mid"] * 100
                if cost <= max_cost:
                    leaps_enriched.append(data)

        if not leaps_enriched:
            self._log_skip(f"No affordable LEAPS with target delta for {symbol}", {"symbol": symbol})
            return None

        # Pick the LEAPS with highest delta (most stock-like behavior)
        leaps_enriched.sort(key=lambda x: abs(x["delta"]), reverse=True)
        best_leaps = leaps_enriched[0]

        # === Short Call Leg (near-term OTM) ===
        short_min_exp = today + timedelta(days=self.PMCC_SHORT_MIN_DAYS)
        short_max_exp = today + timedelta(days=self.PMCC_SHORT_MAX_DAYS)
        short_min_strike = round(underlying_price * 1.03, 2)  # At least 3% OTM
        short_max_strike = round(underlying_price * 1.15, 2)

        short_calls = self.get_option_chain(
            symbol, ContractType.CALL, short_min_exp, short_max_exp,
            min_strike=short_min_strike, max_strike=short_max_strike, limit=50
        )

        if not short_calls:
            self._log_skip(f"No short-term OTM calls for PMCC on {symbol}", {"symbol": symbol})
            return None

        # Enrich short call candidates
        short_enriched = []
        for c in short_calls:
            data = self.enrich_option(c, underlying_price)
            if (data and data["open_interest"] >= self.BCS_MIN_OPEN_INTEREST
                    and data["delta"] is not None
                    and self.PMCC_SHORT_TARGET_DELTA[0] <= abs(data["delta"]) <= self.PMCC_SHORT_TARGET_DELTA[1]
                    and data["mid"] > 0):
                short_enriched.append(data)

        if not short_enriched:
            self._log_skip(f"No suitable short calls for PMCC on {symbol}", {"symbol": symbol})
            return None

        # Pick short call that balances premium vs. upside room
        short_enriched.sort(key=lambda x: x["mid"], reverse=True)
        best_short = short_enriched[0]

        # Calculate PMCC economics
        leaps_cost = best_leaps["mid"] * 100
        short_premium = best_short["mid"] * 100
        net_cost = leaps_cost - short_premium

        # PMCC rule: short call strike should be > LEAPS strike + net debit
        net_debit_per_share = net_cost / 100
        safe = best_short["strike"] > (best_leaps["strike"] + net_debit_per_share)

        pmcc = {
            "strategy": "pmcc",
            "symbol": symbol,
            "underlying_price": underlying_price,
            "leaps_call": best_leaps,
            "short_call": best_short,
            "leaps_cost": round(leaps_cost, 2),
            "short_premium": round(short_premium, 2),
            "net_cost": round(net_cost, 2),
            "capital_efficiency": round(underlying_price * 100 / net_cost, 2) if net_cost > 0 else 0,
            "safe_if_called": safe,
            "monthly_income_potential": round(short_premium, 2),  # Approximate
        }

        logger.info(f"    ✅ Found PMCC: LEAPS ${best_leaps['strike']}C ({best_leaps['dte']}d) + "
                     f"Short ${best_short['strike']}C ({best_short['dte']}d) | "
                     f"Net cost: ${net_cost:.2f} | "
                     f"{'SAFE' if safe else '⚠️ RISK'} if called")

        return pmcc

    def execute_pmcc(self, pmcc: Dict) -> Dict:
        """Execute PMCC by buying LEAPS and selling near-term call."""
        leaps_sym = pmcc["leaps_call"]["symbol"]
        short_sym = pmcc["short_call"]["symbol"]

        logger.info(f"  📥 Executing PMCC: Buy LEAPS {leaps_sym} / Sell {short_sym}")

        if self.dry_run:
            result = {"success": True, "dry_run": True, **pmcc}
            self._log_action("pmcc", result)
            return result

        try:
            order_legs = [
                OptionLegRequest(
                    symbol=leaps_sym,
                    side=OrderSide.BUY,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=short_sym,
                    side=OrderSide.SELL,
                    ratio_qty=1
                )
            ]

            req = MarketOrderRequest(
                qty=1,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=order_legs
            )
            res = self.trading_client.submit_order(req)
            order_id = str(res.id)

            result = {
                "success": True,
                "order_id": order_id,
                "status": res.status.value if hasattr(res.status, 'value') else str(res.status),
                **pmcc
            }
            self._log_action("pmcc", result)
            logger.info(f"    ✅ PMCC order submitted: {order_id}")
            return result

        except Exception as e:
            self._log_error(f"PMCC execution failed for {pmcc['symbol']}: {e}")
            return {"success": False, "error": str(e), **pmcc}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 3: CASH-SECURED PUTS
    # ══════════════════════════════════════════════════════════════════════════

    def find_cash_secured_put(self, symbol: str, underlying_price: float,
                               max_cash_commitment: float) -> Optional[Dict]:
        """
        Find optimal cash-secured put for income + discounted entry.
        """
        logger.info(f"  🔍 Scanning cash-secured puts for {symbol} @ ${underlying_price:.2f}...")

        today = date.today()
        min_exp = today + timedelta(days=self.CSP_EXPIRATION_MIN_DAYS)
        max_exp = today + timedelta(days=self.CSP_EXPIRATION_MAX_DAYS)

        # Strikes below current price
        target_strike = round(underlying_price * (1 - self.CSP_DISCOUNT_TARGET_PCT / 100), 2)
        min_strike = round(underlying_price * 0.80, 2)
        max_strike = round(underlying_price * 0.98, 2)

        # Cash required = strike * 100
        if target_strike * 100 > max_cash_commitment:
            self._log_skip(f"CSP on {symbol} too expensive — need ${target_strike * 100:,.0f}", {"symbol": symbol})
            return None

        puts = self.get_option_chain(symbol, ContractType.PUT, min_exp, max_exp,
                                      min_strike, max_strike)
        if not puts:
            self._log_skip(f"No put options found for {symbol}", {"symbol": symbol})
            return None

        # Enrich and filter
        enriched = []
        for p in puts:
            data = self.enrich_option(p, underlying_price)
            if (data and data["open_interest"] >= self.CSP_MIN_OPEN_INTEREST
                    and data["delta"] is not None
                    and self.CSP_TARGET_DELTA[0] <= abs(data["delta"]) <= self.CSP_TARGET_DELTA[1]
                    and data["mid"] > 0):
                # Check premium yield
                premium_yield = (data["mid"] * 100) / (data["strike"] * 100) * 100
                annual_yield = premium_yield * (365 / max(data["dte"], 1))
                if premium_yield >= self.CSP_MIN_PREMIUM_PCT:
                    data["premium_yield"] = round(premium_yield, 2)
                    data["annual_yield"] = round(annual_yield, 2)
                    enriched.append(data)

        if not enriched:
            self._log_skip(f"No CSP candidates meeting criteria for {symbol}", {"symbol": symbol})
            return None

        # Pick highest yield CSP
        enriched.sort(key=lambda x: x["annual_yield"], reverse=True)
        best = enriched[0]

        cash_required = best["strike"] * 100
        premium_collected = best["mid"] * 100
        effective_entry = best["strike"] - best["mid"]
        discount = ((underlying_price - effective_entry) / underlying_price) * 100

        csp = {
            "strategy": "cash_secured_put",
            "symbol": symbol,
            "underlying_price": underlying_price,
            "put_option": best,
            "cash_required": round(cash_required, 2),
            "premium_collected": round(premium_collected, 2),
            "effective_entry_price": round(effective_entry, 2),
            "discount_to_market": round(discount, 2),
            "premium_yield": best["premium_yield"],
            "annualized_yield": best["annual_yield"],
            "expiration": best["expiration_date"],
            "dte": best["dte"],
        }

        logger.info(f"    ✅ Found CSP: Sell ${best['strike']}P ({best['dte']}d) | "
                     f"Premium: ${premium_collected:.2f} | "
                     f"Yield: {best['premium_yield']:.1f}% ({best['annual_yield']:.0f}% ann.) | "
                     f"Effective entry: ${effective_entry:.2f} ({discount:.1f}% discount)")

        return csp

    def execute_cash_secured_put(self, csp: Dict) -> Dict:
        """Execute a cash-secured put order."""
        put_sym = csp["put_option"]["symbol"]

        logger.info(f"  📥 Executing CSP: Sell {put_sym}")

        if self.dry_run:
            result = {"success": True, "dry_run": True, **csp}
            self._log_action("cash_secured_put", result)
            return result

        try:
            req = MarketOrderRequest(
                symbol=put_sym,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            res = self.trading_client.submit_order(req)
            order_id = str(res.id)

            result = {
                "success": True,
                "order_id": order_id,
                "status": res.status.value if hasattr(res.status, 'value') else str(res.status),
                **csp
            }
            self._log_action("cash_secured_put", result)
            logger.info(f"    ✅ CSP order submitted: {order_id}")
            return result

        except Exception as e:
            self._log_error(f"CSP execution failed for {csp['symbol']}: {e}")
            return {"success": False, "error": str(e), **csp}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 4: PROTECTIVE PUTS
    # ══════════════════════════════════════════════════════════════════════════

    def find_protective_put(self, symbol: str, position: Dict) -> Optional[Dict]:
        """
        Find a protective put for an existing stock position.
        """
        current_price = position.get("current_price", 0)
        market_value = position.get("market_value", 0)
        qty = position.get("qty", 0)

        if market_value < self.PROTECT_MIN_POSITION_VALUE:
            self._log_skip(f"Position {symbol} too small to protect (${market_value:.2f})",
                           {"symbol": symbol})
            return None

        logger.info(f"  🛡️  Scanning protective puts for {symbol} ({qty} shares @ ${current_price:.2f})...")

        today = date.today()
        min_exp = today + timedelta(days=self.PROTECT_EXPIRATION_MIN_DAYS)
        max_exp = today + timedelta(days=self.PROTECT_EXPIRATION_MAX_DAYS)

        target_strike = round(current_price * (1 - self.PROTECT_STRIKE_PCT_BELOW / 100), 2)
        min_strike = round(current_price * 0.80, 2)
        max_strike = round(current_price * 0.95, 2)

        puts = self.get_option_chain(symbol, ContractType.PUT, min_exp, max_exp,
                                      min_strike, max_strike)
        if not puts:
            self._log_skip(f"No protective puts available for {symbol}", {"symbol": symbol})
            return None

        # How many contracts needed? 1 put = 100 shares protection
        contracts_needed = max(1, int(qty / 100))

        # Enrich and filter
        enriched = []
        for p in puts:
            data = self.enrich_option(p, current_price)
            if (data and data["open_interest"] >= 20 and data["mid"] > 0):
                total_cost = data["mid"] * 100 * contracts_needed
                cost_pct = (total_cost / market_value) * 100
                if cost_pct <= self.PROTECT_MAX_COST_PCT:
                    data["total_cost"] = round(total_cost, 2)
                    data["cost_pct"] = round(cost_pct, 2)
                    data["contracts_needed"] = contracts_needed
                    enriched.append(data)

        if not enriched:
            self._log_skip(f"No affordable protective puts for {symbol}", {"symbol": symbol})
            return None

        enriched.sort(key=lambda x: abs(x["strike"] - target_strike))
        best = enriched[0]

        max_loss_per_share = current_price - best["strike"]
        max_loss_pct = (max_loss_per_share / current_price) * 100

        protection = {
            "strategy": "protective_put",
            "symbol": symbol,
            "underlying_price": current_price,
            "position_qty": qty,
            "position_value": round(market_value, 2),
            "put_option": best,
            "contracts": best["contracts_needed"],
            "total_cost": best["total_cost"],
            "cost_as_pct_of_position": best["cost_pct"],
            "max_loss_per_share": round(max_loss_per_share, 2),
            "max_loss_pct": round(max_loss_pct, 2),
            "protection_floor": best["strike"],
            "expiration": best["expiration_date"],
            "dte": best["dte"],
        }

        logger.info(f"    ✅ Found protection: Buy {best['contracts_needed']}x ${best['strike']}P ({best['dte']}d) | "
                     f"Cost: ${best['total_cost']:.2f} ({best['cost_pct']:.1f}% of position) | "
                     f"Max loss capped at {max_loss_pct:.1f}%")

        return protection

    def execute_protective_put(self, protection: Dict) -> Dict:
        """Execute a protective put purchase."""
        put_sym = protection["put_option"]["symbol"]
        contracts = protection["contracts"]

        logger.info(f"  📥 Buying {contracts}x {put_sym} for protection")

        if self.dry_run:
            result = {"success": True, "dry_run": True, **protection}
            self._log_action("protective_put", result)
            return result

        try:
            req = MarketOrderRequest(
                symbol=put_sym,
                qty=contracts,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            res = self.trading_client.submit_order(req)
            order_id = str(res.id)

            result = {
                "success": True,
                "order_id": order_id,
                "status": res.status.value if hasattr(res.status, 'value') else str(res.status),
                **protection
            }
            self._log_action("protective_put", result)
            logger.info(f"    ✅ Protective put order submitted: {order_id}")
            return result

        except Exception as e:
            self._log_error(f"Protective put failed for {protection['symbol']}: {e}")
            return {"success": False, "error": str(e), **protection}

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 5: ZERO-COST COLLARS (NEW)
    # ══════════════════════════════════════════════════════════════════════════

    def find_zero_cost_collar(self, symbol: str, position: Dict) -> Optional[Dict]:
        """
        Buys an OTM Put, Sells an OTM Call to offset cost. Needs 100+ shares.
        """
        price = position.get("current_price", self.get_current_price(symbol))
        qty = position.get("qty", 0)
        value = position.get("market_value", 0)
        
        if value < self.PROTECT_MIN_POSITION_VALUE or qty < 100: 
            return None

        logger.info(f"  👔 Structuring Zero-Cost Collar for {symbol}...")
        contracts_needed = int(qty // 100)
        min_exp = date.today() + timedelta(days=self.PROTECT_EXPIRATION_MIN_DAYS)
        max_exp = date.today() + timedelta(days=self.PROTECT_EXPIRATION_MAX_DAYS)

        # 1. Fetch Puts for protection
        put_target = round(price * (1 - self.PROTECT_STRIKE_PCT_BELOW / 100), 2)
        puts = self.get_option_chain(symbol, ContractType.PUT, min_exp, max_exp, max_strike=put_target*1.05)
        enriched_puts = [d for p in puts if (d := self.enrich_option(p, price))]
        if not enriched_puts: return None
        best_put = sorted(enriched_puts, key=lambda x: abs(x["strike"] - put_target))[0]

        # 2. Fetch Calls at the exact same expiration to fund the Put
        call_target = round(price * (1 + self.COLLAR_CALL_STRIKE_PCT_ABOVE / 100), 2)
        exp_date_obj = datetime.strptime(best_put["expiration_date"], "%Y-%m-%d").date()
        calls = self.get_option_chain(symbol, ContractType.CALL, exp_date_obj, exp_date_obj, min_strike=call_target*0.9)
        enriched_calls = [d for c in calls if (d := self.enrich_option(c, price)) and d["mid"] >= (best_put["mid"] * 0.8)]
        if not enriched_calls: return None
        
        # 3. Match prices to achieve Zero Cost
        put_cost = best_put["ask"]
        best_call, smallest_net_cost = None, float('inf')

        for call in enriched_calls:
            net_cost = put_cost - call["bid"]
            if -0.50 <= net_cost <= 0.20: # We accept paying up to 20 cents per share for the collar
                if net_cost < smallest_net_cost:
                    smallest_net_cost = net_cost
                    best_call = call

        if not best_call: return None

        net_total = smallest_net_cost * 100 * contracts_needed
        logger.info(f"    ✅ Collar Found: Buy ${best_put['strike']}P / Sell ${best_call['strike']}C | Net Cost: ${net_total:.2f}")

        return {
            "strategy": "zero_cost_collar", 
            "symbol": symbol, 
            "contracts": contracts_needed,
            "put_leg": best_put, 
            "call_leg": best_call, 
            "net_cost": round(net_total, 2)
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STRATEGY 6: MACRO HEDGE (NEW)
    # ══════════════════════════════════════════════════════════════════════════

    def find_macro_hedge(self, account: Dict) -> Optional[Dict]:
        """If regime is Bearish, use 1% of portfolio equity to buy SPY Puts."""
        hedge_budget = account.get("equity", 0) * (self.MACRO_HEDGE_ALLOCATION_PCT / 100)
        if hedge_budget < 100: return None

        spy_price = self.get_current_price(self.MACRO_HEDGE_TICKER)
        if not spy_price: return None

        logger.info(f"  📉 Structuring Macro Hedge (SPY Puts). Budget: ${hedge_budget:,.2f}")
        min_exp = date.today() + timedelta(days=self.MACRO_HEDGE_DTE)
        max_exp = date.today() + timedelta(days=90)
        target_strike = round(spy_price * 0.90, 2)

        puts = self.get_option_chain(self.MACRO_HEDGE_TICKER, ContractType.PUT, min_exp, max_exp, max_strike=target_strike*1.05)
        enriched = [d for p in puts if (d := self.enrich_option(p, spy_price))]
        if not enriched: return None
        
        best_put = sorted(enriched, key=lambda x: abs(x["strike"] - target_strike))[0]
        put_cost = best_put["ask"] * 100
        contracts = max(1, int(hedge_budget // put_cost))

        if contracts * put_cost > account.get("cash", 0): return None

        logger.info(f"    ✅ Macro Hedge: Buy {contracts}x {self.MACRO_HEDGE_TICKER} ${best_put['strike']}P | Total: ${contracts * put_cost:.2f}")
        return {
            "strategy": "macro_hedge", 
            "symbol": self.MACRO_HEDGE_TICKER, 
            "contracts": contracts, 
            "put_option": best_put, 
            "total_cost": round(contracts * put_cost, 2)
        }

    def execute_multi_leg(self, data: Dict) -> Dict:
        """Executes multi-leg options like collars, or single options like macro hedge."""
        strat = data["strategy"]
        logger.info(f"  📥 Executing {strat.replace('_', ' ').title()} for {data.get('symbol')}")

        if self.dry_run:
            self.execution_log["trades_executed"].append({"status": "DRY_RUN", **data})
            return {"success": True, "dry_run": True, **data}
        
        try:
            if strat == "zero_cost_collar":
                legs = [
                    OptionLegRequest(symbol=data["put_leg"]["symbol"], side=OrderSide.BUY, ratio_qty=1),
                    OptionLegRequest(symbol=data["call_leg"]["symbol"], side=OrderSide.SELL, ratio_qty=1)
                ]
                req = MarketOrderRequest(qty=data["contracts"], order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY, legs=legs)
            elif strat == "macro_hedge":
                sym = data.get("put_option")["symbol"]
                req = MarketOrderRequest(symbol=sym, qty=data["contracts"], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                
            res = self.trading_client.submit_order(req)
            data["order_id"] = str(res.id)
            self.execution_log["trades_executed"].append({"status": "EXECUTED", "order_id": str(res.id), **data})
            logger.info(f"    ✅ Order submitted: {res.id}")
            return {"success": True, **data}
        except Exception as e:
            logger.error(f"    ❌ Execution failed: {e}")
            self._log_error(f"Execution failed: {e}")
            return {"success": False, "error": str(e)}


    # ══════════════════════════════════════════════════════════════════════════
    # MASTER ORCHESTRATOR
    # ══════════════════════════════════════════════════════════════════════════

    def run_all_strategies(self, strategies: List[str] = None):
        """
        Run the complete options strategy pipeline based on Volatility Rank routing.
        """
        all_strategies = ["spreads", "pmcc", "csp", "protect", "collar", "macro"]
        active = strategies or all_strategies

        print("\n" + "═" * 70)
        print("  🎯 INSTITUTIONAL OPTIONS OVERLAY (Casino Edge & Collars)")
        print(f"  Mode: {'DRY RUN' if self.dry_run else '🔴 LIVE'}")
        print(f"  Active Strategies: {', '.join(active)}")
        print("═" * 70)

        ok, account = self.preflight_check()
        if not ok: return self.execution_log

        budget = self.calculate_max_options_budget(account)
        budget_remaining = budget
        max_per_trade = self.max_trade_size(account)

        recommendations = self.load_recommendations() or {}
        positions = self.get_positions()

        buy_signals = []
        for sig in recommendations.get("buy_signals", []):
            if "/" not in sig.get("symbol", "") and sig.get("asset_type") != "crypto":
                buy_signals.append(sig)
        buy_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        results = {"bull_call_spreads": [], "pmcc": [], "cash_secured_puts": [], "protective_puts": [], "collars": [], "macro_hedge": []}

        # ── 1. IV RANK ROUTING (Alpha Generation) ──
        if any(s in active for s in ["spreads", "pmcc", "csp"]):
            print(f"\n{'─' * 70}")
            print("  🎰 AI CASINO MODE: Analyzing IV Rank (Volatility) for Routing...")
            print(f"{'─' * 70}")

            csp_cash_used = 0
            max_csp_cash = account.get("cash", 0) * (self.CSP_CASH_RESERVE_PCT / 100)

            for cand in buy_signals[:8]:
                if budget_remaining < 100 and csp_cash_used >= max_csp_cash: 
                    logger.info("  💰 Options and CSP budget exhausted")
                    break
                
                sym = cand["symbol"]
                conf = cand["confidence"]
                price = self.get_current_price(sym)
                if not price: continue

                iv_rank = self.get_volatility_rank(sym)
                print(f"\n  ► {sym:<6} | Conf: {conf:.0%} | Price: ${price:.2f}")
                print(f"    IV Rank: {iv_rank:.1f}/100 -> ", end="")

                if iv_rank > 60:
                    print("Opciones CARAS. Casino Mode: VENDER Primas (CSP)")
                    if "csp" in active and conf >= self.CSP_MIN_CONFIDENCE and sym not in positions:
                        if "cash_secured_puts" not in self.execution_log["strategies_run"]: 
                            self.execution_log["strategies_run"].append("cash_secured_puts")
                        
                        if max_csp_cash - csp_cash_used > 1000:
                            csp = self.find_cash_secured_put(sym, price, max_csp_cash - csp_cash_used)
                            if csp:
                                res = self.execute_cash_secured_put(csp)
                                results["cash_secured_puts"].append(res)
                                if res.get("success"): 
                                    csp_cash_used += csp["cash_required"]
                else:
                    print("Opciones BARATAS. Buyer Mode: COMPRAR Primas (Spread/PMCC)")
                    pmcc_executed = False
                    if "pmcc" in active and conf >= self.PMCC_MIN_CONFIDENCE and budget_remaining > max_per_trade * 0.5:
                        if "pmcc" not in self.execution_log["strategies_run"]: 
                            self.execution_log["strategies_run"].append("pmcc")
                        pmcc = self.find_pmcc(sym, price, min(max_per_trade, budget_remaining))
                        if pmcc:
                            res = self.execute_pmcc(pmcc)
                            results["pmcc"].append(res)
                            if res.get("success"): 
                                budget_remaining -= pmcc["net_cost"]
                                pmcc_executed = True
                    
                    if not pmcc_executed and "spreads" in active and conf >= self.BCS_MIN_CONFIDENCE and budget_remaining > max_per_trade * 0.3:
                        if "bull_call_spreads" not in self.execution_log["strategies_run"]: 
                            self.execution_log["strategies_run"].append("bull_call_spreads")
                        spread = self.find_bull_call_spread(sym, price, min(max_per_trade, budget_remaining))
                        if spread:
                            res = self.execute_bull_call_spread(spread)
                            results["bull_call_spreads"].append(res)
                            if res.get("success"): 
                                budget_remaining -= spread["net_debit"]

        # ── 2. PORTFOLIO PROTECTION (Collars & Puts) ──
        if "collar" in active or "protect" in active:
            print(f"\n{'─' * 70}")
            print("  🛡️  Evaluating Zero-Cost Collars & Protective Puts on existing holdings")
            print(f"{'─' * 70}")
            for sym, pos in positions.items():
                if "equity" in pos["asset_class"].lower() and not self._is_option_symbol(sym):
                    col = None
                    if "collar" in active:
                        col = self.find_zero_cost_collar(sym, pos)
                        if col:
                            if "collars" not in self.execution_log["strategies_run"]: 
                                self.execution_log["strategies_run"].append("collars")
                            res = self.execute_multi_leg(col)
                            results["collars"].append(res)
                    
                    if not col and "protect" in active and budget_remaining > 100:
                        prot = self.find_protective_put(sym, pos)
                        if prot:
                            if "protective_puts" not in self.execution_log["strategies_run"]: 
                                self.execution_log["strategies_run"].append("protective_puts")
                            res = self.execute_protective_put(prot)
                            results["protective_puts"].append(res)
                            if res.get("success"): 
                                budget_remaining -= prot["total_cost"]

        # ── 3. MACRO HEDGE ──
        regime = "sideways"
        if os.path.exists(os.path.join(self.data_dir, "regime_context.json")):
            try: 
                with open(os.path.join(self.data_dir, "regime_context.json"), 'r') as f:
                    regime_data = json.load(f)
                    regime = regime_data.get("regime", "sideways").lower()
            except: pass

        if "macro" in active and "bear" in regime:
            print(f"\n{'─' * 70}")
            print("  📉 BEAR REGIME DETECTED: Deploying Macro Hedge")
            print(f"{'─' * 70}")
            hedge = self.find_macro_hedge(account)
            if hedge:
                if "macro_hedge" not in self.execution_log["strategies_run"]: 
                    self.execution_log["strategies_run"].append("macro_hedge")
                res = self.execute_multi_leg(hedge)
                results["macro_hedge"].append(res)

        self._print_summary(results, budget, budget_remaining)
        self._save_results(results)
        return self.execution_log

    def _print_summary(self, results: Dict, initial_budget: float, remaining_budget: float):
        print(f"\n{'═' * 70}")
        print("  📊 OPTIONS ENGINE SUMMARY")
        print(f"{'═' * 70}")

        total_trades = 0
        total_capital = initial_budget - remaining_budget

        for strategy, trades in results.items():
            successful = [t for t in trades if t.get("success")]
            total_trades += len(successful)

            strategy_name = strategy.replace("_", " ").title()
            if successful:
                print(f"\n  {strategy_name}: {len(successful)} trade(s)")
                for t in successful:
                    sym = t.get("symbol", "?")
                    if strategy == "bull_call_spreads":
                        cost = t.get("net_debit", 0)
                        max_p = t.get("max_profit", 0)
                        print(f"    • {sym}: Cost ${cost:.2f} → Max profit ${max_p:.2f}")
                    elif strategy == "pmcc":
                        cost = t.get("net_cost", 0)
                        print(f"    • {sym}: Net cost ${cost:.2f}")
                    elif strategy == "cash_secured_puts":
                        prem = t.get("premium_collected", 0)
                        yld = t.get("annualized_yield", 0)
                        print(f"    • {sym}: Premium ${prem:.2f} ({yld:.0f}% annualized)")
                    elif strategy == "protective_puts":
                        cost = t.get("total_cost", 0)
                        floor_pct = t.get("max_loss_pct", 0)
                        print(f"    • {sym}: Cost ${cost:.2f} (loss capped at {floor_pct:.1f}%)")
                    elif strategy == "collars":
                        cost = t.get("net_cost", 0)
                        print(f"    • {sym}: Zero-Cost Collar Executed (Net Cost: ${cost:.2f})")
                    elif strategy == "macro_hedge":
                        cost = t.get("total_cost", 0)
                        print(f"    • {sym}: Macro Hedge Placed (Total Cost: ${cost:.2f})")
            else:
                pass

        print(f"\n  {'─' * 66}")
        print(f"  Total trades: {total_trades}")
        print(f"  Capital deployed: ${total_capital:,.2f}")
        print(f"  Budget remaining: ${remaining_budget:,.2f}")
        print(f"  Mode: {'DRY RUN — no real orders placed' if self.dry_run else '🔴 LIVE'}")
        print(f"{'═' * 70}\n")

        self.execution_log["summary"] = {
            "total_trades": total_trades,
            "capital_deployed": round(total_capital, 2),
            "budget_remaining": round(remaining_budget, 2)
        }

    def _save_results(self, results: Dict):
        """Save execution results to file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "mode": "DRY RUN" if self.dry_run else "LIVE",
            "results": {},
            "log": self.execution_log
        }

        for strategy, trades in results.items():
            output["results"][strategy] = []
            for trade in trades:
                clean = {}
                for k, v in trade.items():
                    if isinstance(v, (str, int, float, bool, type(None), list)):
                        clean[k] = v
                    elif isinstance(v, dict):
                        clean[k] = {kk: vv for kk, vv in v.items()
                                     if isinstance(vv, (str, int, float, bool, type(None)))}
                    else:
                        clean[k] = str(v)
                output["results"][strategy].append(clean)

        filepath = os.path.join(self.data_dir, "options_trades.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
        except Exception as e:
            self._log_error(f"Failed to save results: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy", nargs="+", choices=["spreads", "pmcc", "csp", "protect", "collar", "macro"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.force:
        confirm = input("\n⚠️  LIVE OPTIONS TRADING MODE. Type 'yes' to continue: ")
        if confirm.lower() != "yes":
            return 0

    engine = OptionsStrategyEngine(dry_run=args.dry_run)
    engine.run_all_strategies(strategies=args.strategy)
    return 0

if __name__ == "__main__":
    sys.exit(main())