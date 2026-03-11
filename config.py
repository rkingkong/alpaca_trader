#!/usr/bin/env python3
"""
Shared Configuration & Client Factory
=======================================
Single source of truth for all API credentials, client initialization,
regime context management, logging infrastructure, and risk constants.

Every module imports from here — no duplicated config loading.

Institutional Enhancements:
  - Centralized logging (file + console, rotated daily)
  - Max drawdown tracking with circuit breaker
  - Sector concentration limits
  - Rate-limited API helper
  - Market hours awareness
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from zoneinfo import ZoneInfo

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
SNAPSHOTS_DIR = os.path.join(SCRIPT_DIR, "snapshots")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# Ensure directories exist
for d in (DATA_DIR, LOGS_DIR, SNAPSHOTS_DIR):
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOGGING INFRASTRUCTURE
# =============================================================================

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_log_initialized = False


def setup_logging(name: str = "luxverum", level: int = logging.INFO) -> logging.Logger:
    """
    Set up centralized logging with both console and daily-rotated file output.

    Call once per process (idempotent). All modules should call:
        logger = setup_logging(__name__)
    """
    global _log_initialized

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _log_initialized:
        # Root logger config — applies to all child loggers
        root = logging.getLogger()
        root.setLevel(level)

        # Guard: only add handlers if root doesn't already have them
        # (prevents duplication if setup_logging is called multiple times
        # in the same process, e.g., via different import paths)
        if not root.handlers:
            # Console handler (INFO+)
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level)
            console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
            root.addHandler(console)

            # File handler — one log per day, keep 30 days
            log_file = os.path.join(LOGS_DIR, "trading_system.log")
            try:
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file, when="midnight", backupCount=30, encoding="utf-8"
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FMT))
                root.addHandler(file_handler)
            except (PermissionError, OSError):
                pass  # File may be locked by another process

        _log_initialized = True

    return logger


# =============================================================================
# CONFIG LOADING (single source of truth)
# =============================================================================

class Config:
    """Centralized configuration loaded once, used everywhere."""

    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not Config._loaded:
            self._load()
            Config._loaded = True

    def _load(self):
        """Load from config.json with env var fallbacks."""
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)

        def get(key):
            return cfg.get(key) or os.getenv(key)

        self.ALPACA_API_KEY = get('ALPACA_API_KEY')
        self.ALPACA_SECRET_KEY = get('ALPACA_SECRET_KEY')
        self.ANTHROPIC_API_KEY = get('ANTHROPIC_API_KEY')
        self.OPENAI_API_KEY = get('OPENAI_API_KEY')
        self.NEWSAPI_KEY = get('NEWSAPI_KEY')
        self.FINNHUB_API_KEY = get('FINNHUB_API_KEY')
        self.FRED_API_KEY = get('FRED_API_KEY')
        self.ALPHA_VANTAGE_KEY = get('ALPHA_VANTAGE_KEY')
        self.PAPER_TRADING = cfg.get('PAPER_TRADING', True)

    def validate(self, require_ai=False, require_trading=True) -> bool:
        """Validate that required credentials are present."""
        ok = True
        if require_trading:
            if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
                print("  FATAL: Alpaca API credentials missing")
                ok = False
        if require_ai:
            if not self.ANTHROPIC_API_KEY:
                print("  WARNING: Anthropic API key missing")
            if not self.OPENAI_API_KEY:
                print("  WARNING: OpenAI API key missing")
        return ok


# =============================================================================
# CLIENT FACTORY (lazy initialization, shared instances)
# =============================================================================

class Clients:
    """Lazy-initialized API client factory. Creates clients on first access."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._trading = None
            cls._instance._stock_data = None
            cls._instance._crypto_data = None
            cls._instance._option_data = None
            cls._instance._claude = None
            cls._instance._openai = None
        return cls._instance

    @property
    def config(self):
        return Config()

    @property
    def trading(self):
        if self._trading is None:
            from alpaca.trading.client import TradingClient
            self._trading = TradingClient(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY,
                paper=self.config.PAPER_TRADING
            )
        return self._trading

    @property
    def stock_data(self):
        if self._stock_data is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._stock_data = StockHistoricalDataClient(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY
            )
        return self._stock_data

    @property
    def crypto_data(self):
        if self._crypto_data is None:
            from alpaca.data.historical import CryptoHistoricalDataClient
            self._crypto_data = CryptoHistoricalDataClient(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY
            )
        return self._crypto_data

    @property
    def option_data(self):
        if self._option_data is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._option_data = OptionHistoricalDataClient(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY
            )
        return self._option_data

    @property
    def claude(self):
        if self._claude is None and self.config.ANTHROPIC_API_KEY:
            import anthropic
            self._claude = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
        return self._claude

    @property
    def openai(self):
        if self._openai is None and self.config.OPENAI_API_KEY:
            from openai import OpenAI
            self._openai = OpenAI(api_key=self.config.OPENAI_API_KEY)
        return self._openai


# =============================================================================
# REGIME CONTEXT (shared across modules)
# =============================================================================

@dataclass
class RegimeContext:
    """Market regime context -- shared data contract between modules."""
    regime: str = "sideways"
    regime_score: float = 0.0
    trend_strength: float = 50.0
    volatility_regime: str = "normal"
    risk_appetite: float = 0.50
    sector_rankings: Dict = None
    sector_momentum: Dict = None
    recommended_position_size_pct: float = 3.5
    recommended_max_positions: int = 15
    recommended_stop_loss_pct: float = 5.0
    recommended_take_profit_pct: float = 12.0
    recommended_cash_reserve_pct: float = 10.0
    recommended_confidence_threshold: float = 0.55
    spy_trend: str = "flat"
    spy_momentum_1m: float = 0.0
    spy_momentum_3m: float = 0.0
    spy_above_sma50: bool = True
    spy_above_sma200: bool = True
    golden_cross: bool = True
    timestamp: str = ""

    def __post_init__(self):
        if self.sector_rankings is None:
            self.sector_rankings = {}
        if self.sector_momentum is None:
            self.sector_momentum = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def save_regime_context(ctx: RegimeContext):
    """Save regime context to shared data directory."""
    filepath = os.path.join(DATA_DIR, "regime_context.json")
    with open(filepath, 'w') as f:
        json.dump(asdict(ctx), f, indent=2)


def load_regime_context(max_age_seconds: int = 7200) -> Optional[RegimeContext]:
    """Load regime context if fresh enough (default: 2 hours)."""
    filepath = os.path.join(DATA_DIR, "regime_context.json")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check freshness
        ts = data.get("timestamp", "")
        if ts and max_age_seconds > 0:
            try:
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
                if (now - parsed).total_seconds() > max_age_seconds:
                    return None
            except Exception:
                pass  # If parsing fails, use it anyway

        return RegimeContext(**data)
    except Exception:
        return None


# =============================================================================
# MARKET HOURS & TRADING CALENDAR
# =============================================================================

NY_TZ = ZoneInfo("America/New_York")


def is_market_hours() -> bool:
    """Check if US stock market is currently open."""
    now = datetime.now(NY_TZ)
    # Weekday check (0=Mon, 4=Fri)
    if now.weekday() > 4:
        return False
    from datetime import time as dt_time
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    return market_open <= now.time() <= market_close


def is_extended_hours() -> bool:
    """Check if we are in pre-market (4am-9:30am) or after-hours (4pm-8pm)."""
    now = datetime.now(NY_TZ)
    if now.weekday() > 4:
        return False
    from datetime import time as dt_time
    pre_open = dt_time(4, 0)
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    after_close = dt_time(20, 0)
    t = now.time()
    return (pre_open <= t < market_open) or (market_close < t <= after_close)


def market_status() -> str:
    """Return human-readable market status."""
    if is_market_hours():
        return "OPEN"
    if is_extended_hours():
        return "EXTENDED_HOURS"
    return "CLOSED"


# =============================================================================
# SECTOR CONCENTRATION LIMITS
# =============================================================================

# Maps each stock to its GICS sector for concentration tracking.
# This is populated at runtime by the analysis module, but we define
# the limits here so every module can reference them.

MAX_SECTOR_CONCENTRATION_PCT = 30.0   # No single sector > 30% of portfolio
MAX_SINGLE_POSITION_PCT = 10.0        # Hard cap on any one position
MAX_CORRELATED_BLOCK_PCT = 40.0       # Correlated group (e.g., all crypto) cap

# Sector mapping for well-known symbols (extended at runtime)
SYMBOL_SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "SMCI": "Technology", "PLTR": "Technology",
    "TSM": "Technology", "AVGO": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "MRNA": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    "ISRG": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    # Financial
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
    "V": "Financial", "MA": "Financial", "COIN": "Financial",
    "SOFI": "Financial", "BRK.B": "Financial", "MS": "Financial",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
    "FSLR": "Energy", "CEG": "Energy", "VST": "Energy",
    "ENPH": "Energy", "NEE": "Energy",
    # Consumer
    "AMZN": "Consumer", "WMT": "Consumer", "COST": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "LULU": "Consumer",
    "TGT": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    # Industrial
    "CAT": "Industrial", "DE": "Industrial", "BA": "Industrial",
    "HON": "Industrial", "GE": "Industrial", "RTX": "Industrial",
    "LMT": "Industrial", "UPS": "Industrial",
    # Communications
    "DIS": "Communications", "NFLX": "Communications",
    "T": "Communications", "VZ": "Communications",
    "SPOT": "Communications", "RBLX": "Communications",
    # Real Estate
    "O": "Real Estate", "AMT": "Real Estate", "PLD": "Real Estate",
    "SPG": "Real Estate", "EQIX": "Real Estate", "DLR": "Real Estate",
    # Materials
    "FCX": "Materials", "NEM": "Materials", "LIN": "Materials",
    "APD": "Materials", "NUE": "Materials", "GOLD": "Materials",
    # Crypto (treated as its own sector for concentration)
    "BTC/USD": "Crypto", "BTCUSD": "Crypto",
    "ETH/USD": "Crypto", "ETHUSD": "Crypto",
    "SOL/USD": "Crypto", "SOLUSD": "Crypto",
    "AVAX/USD": "Crypto", "AVAXUSD": "Crypto",
    "DOGE/USD": "Crypto", "DOGEUSD": "Crypto",
    "LTC/USD": "Crypto", "LTCUSD": "Crypto",
    "LINK/USD": "Crypto", "LINKUSD": "Crypto",
    "UNI/USD": "Crypto", "UNIUSD": "Crypto",
    "SHIB/USD": "Crypto", "SHIBUSD": "Crypto",
    # Sector ETFs (map to their corresponding sector)
    "XLK": "Technology", "QQQ": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financial",
    "XLE": "Energy",
    "XLY": "Consumer", "XLP": "Consumer",
    "XLI": "Industrial",
    "XLRE": "Real Estate",
    "XLC": "Communications",
    "XLB": "Materials",
    "XLU": "Utilities",
    # Index / Multi-sector ETFs
    "SPY": "Index", "IWM": "Index", "DIA": "Index",
    # Commodity ETFs
    "GLD": "Commodity", "SLV": "Commodity", "USO": "Commodity",
    "UNG": "Commodity",
    # Additional common holdings
    "TSLA": "Consumer", "MSTR": "Technology",
    "OXY": "Energy", "VLO": "Energy", "HAL": "Energy",
    "LNG": "Energy", "KMI": "Energy", "CEG": "Energy",
    "VST": "Energy", "FSLR": "Energy",
    "SBUX": "Consumer", "TGT": "Consumer", "LULU": "Consumer",
    "LYV": "Communications", "RBLX": "Communications",
    "COIN": "Financial", "SOFI": "Financial",
    "PLTR": "Technology", "SMCI": "Technology", "ARM": "Technology",
    "MRVL": "Technology", "NET": "Technology", "SNOW": "Technology",
    "DDOG": "Technology", "PANW": "Technology", "MDB": "Technology",
    "LLY": "Healthcare", "ISRG": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "MRNA": "Healthcare", "ABBV": "Healthcare",
    "PFE": "Healthcare",
    "RTX": "Industrial", "LMT": "Industrial", "UPS": "Industrial",
    "GE": "Industrial", "HON": "Industrial",
    "GOLD": "Materials", "FCX": "Materials", "NEM": "Materials",
    "NUE": "Materials",
    "O": "Real Estate", "DLR": "Real Estate", "PLD": "Real Estate",
    "AMT": "Real Estate", "SPG": "Real Estate", "EQIX": "Real Estate",
}


def get_sector(symbol: str) -> str:
    """Look up the sector for a symbol. Returns 'Unknown' if not mapped."""
    return SYMBOL_SECTOR_MAP.get(symbol, "Unknown")


# =============================================================================
# MAX DRAWDOWN TRACKER
# =============================================================================

@dataclass
class DrawdownState:
    """Tracks portfolio high-water mark and max drawdown."""
    high_water_mark: float = 0.0
    current_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_date: str = ""
    last_updated: str = ""

    @property
    def current_drawdown_pct(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return round((1 - self.current_equity / self.high_water_mark) * 100, 2)


_DRAWDOWN_FILE = os.path.join(DATA_DIR, "drawdown_state.json")


def load_drawdown_state() -> DrawdownState:
    """Load the drawdown tracker state."""
    try:
        with open(_DRAWDOWN_FILE, 'r') as f:
            data = json.load(f)
        return DrawdownState(**data)
    except Exception:
        return DrawdownState()


def update_drawdown(equity: float) -> DrawdownState:
    """Update drawdown tracker with current equity. Returns state."""
    state = load_drawdown_state()
    state.current_equity = equity
    state.last_updated = datetime.now().isoformat()

    if equity > state.high_water_mark:
        state.high_water_mark = equity

    if state.high_water_mark > 0:
        dd = (1 - equity / state.high_water_mark) * 100
        if dd > state.max_drawdown_pct:
            state.max_drawdown_pct = round(dd, 2)
            state.max_drawdown_date = date.today().isoformat()

    with open(_DRAWDOWN_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2)

    return state


# Circuit breaker threshold
MAX_DRAWDOWN_CIRCUIT_BREAKER_PCT = 15.0  # Halt all new buys if drawdown > 15%


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 3.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        """Block until it is safe to make the next call."""
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()


# Global rate limiter for Alpaca data API (200 calls/min = ~3.3/sec)
alpaca_rate_limiter = RateLimiter(calls_per_second=3.0)


# =============================================================================
# TOP FUND BENCHMARKS
# =============================================================================

TOP_FUND_BENCHMARKS = {
    "elite_hedge_fund_cagr": 20.0,
    "top_mutual_fund_cagr": 13.0,
    "sp500_long_term_cagr": 10.5,
    "bridgewater_pure_alpha_2025": 33.0,
    "discovery_capital_2024": 52.0,
    "our_target_range": 18.0,
}

# Sector ETF mapping (used by regime detector and analysis)
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrial": "XLI",
    "Real Estate": "XLRE",
    "Communications": "XLC",
    "Materials": "XLB",
    "Utilities": "XLU",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_json(filename: str) -> Optional[Dict]:
    """Load a JSON file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_json(filename: str, data, indent: int = 2):
    """Save data to a JSON file in the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    import numpy as np

    def default(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=default)


def recommendations_age_seconds() -> Optional[float]:
    """Return the age in seconds of the recommendations file, or None if missing."""
    filepath = os.path.join(DATA_DIR, "recommendations.json")
    try:
        data = load_json("recommendations.json")
        if data and "timestamp" in data:
            ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
            return (now - ts).total_seconds()
        # Fallback to file mtime
        mtime = os.path.getmtime(filepath)
        return time.time() - mtime
    except Exception:
        return None


MAX_RECOMMENDATIONS_AGE_SECONDS = 14400  # 4 hours — refuse to execute stale signals


def print_header(text: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_step(step_num: int, total: int, text: str):
    """Print a numbered step indicator."""
    print(f"\n{'─' * 70}")
    print(f"  Step {step_num}/{total}: {text}")
    print(f"{'─' * 70}\n")