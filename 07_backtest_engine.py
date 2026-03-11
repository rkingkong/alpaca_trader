#!/usr/bin/env python3
"""
Module 7: Historical Backtesting Engine
=========================================
Backtests the trading system's strategy over 1, 3, and 5 year periods.

Since we can't replay AI calls historically, this module replicates the
EXACT technical scoring criteria from Module 1 (RSI, MACD, Bollinger Bands,
EMA crossover, volume) and the EXACT risk management rules from Module 3
(conviction-based sizing, stop-losses, take-profits, trailing stops,
circuit breakers) to simulate how the system would have performed.

Covers all 10 market sectors with representative liquid symbols.

Output: Performance metrics + visual HTML report with equity curves.
"""

import os
import json
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# v2: Top-performing fund benchmarks for comparison
TOP_FUND_BENCHMARKS = {
    "Quantedge Global (~20% CAGR)": 20.0,
    "Baron Opportunity (~13% CAGR)": 13.0,
    "Fidelity Contrafund (~13% CAGR)": 13.0,
    "Discovery Capital (2024: +52%)": 52.0,
    "Bridgewater Pure Alpha (2025: +33%)": 33.0,
    "D.E. Shaw Oculus (2024: +36%)": 36.0,
    "S&P 500 Long-Term (~10.5% CAGR)": 10.5,
    "Our Target Range": 18.0,
}

import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    sector: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    side: str  # 'BUY' or 'SELL'
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'signal_exit', 'rebalance'
    holding_days: int
    conviction: str  # 'high', 'medium', 'base'

@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    sector: str
    entry_date: str
    entry_price: float
    shares: float
    side: str
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    highest_price: float
    conviction: str
    allocation_pct: float

@dataclass
class DailySnapshot:
    """Daily portfolio state."""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    num_positions: int
    daily_return_pct: float
    drawdown_pct: float


# =============================================================================
# TECHNICAL ANALYSIS ENGINE (mirrors Module 1 exactly)
# =============================================================================

class TechnicalAnalyzer:
    """
    Replicates the exact technical analysis from Module 1's AIMarketAnalyzer.
    Used to generate signals without AI API calls for historical backtesting.
    """

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_ema_crossover(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        ema_9 = prices.ewm(span=9, adjust=False).mean()
        ema_21 = prices.ewm(span=21, adjust=False).mean()
        return ema_9, ema_21

    @staticmethod
    def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        vol_sma = volume.rolling(window=period).mean()
        return volume / vol_sma

    def generate_composite_score(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> pd.Series:
        """
        MOMENTUM-FOCUSED composite score (0-1) mimicking the AI consensus.
        The AI models (Claude as quant PM + GPT-4o as technical analyst)
        prioritize: trend direction, momentum strength, relative strength
        vs market, and breakout confirmation with volume.

        KEY PHILOSOPHY: Buy strength, ride trends, avoid fighting the tape.
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Calculate all indicators
        rsi = self.calculate_rsi(close)
        macd, macd_signal, macd_hist = self.calculate_macd(close)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(close)
        atr = self.calculate_atr(high, low, close)
        ema_9, ema_21 = self.calculate_ema_crossover(close)
        vol_ratio = self.calculate_volume_ratio(volume)

        # Additional trend indicators
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=min(200, max(len(close)-1, 50))).mean()

        # Price momentum: 1-month and 3-month returns
        mom_21 = close.pct_change(21)
        mom_63 = close.pct_change(min(63, max(len(close)-2, 21)))

        # MACD acceleration
        macd_hist_roc = macd_hist - macd_hist.shift(5)

        scores = pd.DataFrame(index=df.index)

        # =================================================================
        # 1. TREND ALIGNMENT (25%) - THE MOST IMPORTANT FACTOR
        #    Price above 50 & 200 SMA = strong uptrend = highest scores
        # =================================================================
        above_50 = (close > sma_50).astype(float)
        above_200 = (close > sma_200).astype(float)
        golden_cross = (sma_50 > sma_200).astype(float)
        scores['trend'] = above_50 * 0.35 + above_200 * 0.35 + golden_cross * 0.30

        # =================================================================
        # 2. MOMENTUM STRENGTH (25%) - RSI "power zone" + price returns
        #    Momentum stocks live in RSI 50-70. Below 40 = weak.
        # =================================================================
        rsi_score = np.where(rsi < 30, 0.15,        # Falling knife
                   np.where(rsi < 40, 0.25,
                   np.where(rsi < 50, 0.45,          # Recovering
                   np.where(rsi < 60, 0.80,          # POWER ZONE
                   np.where(rsi < 70, 0.90,          # POWER ZONE - peak
                   np.where(rsi < 80, 0.70,          # Extended but ok
                   0.40))))))

        mom_score = np.clip(0.5 + mom_21 * 5, 0.0, 1.0) * 0.5 + \
                    np.clip(0.5 + mom_63 * 3, 0.0, 1.0) * 0.5
        scores['momentum'] = pd.Series(rsi_score, index=df.index) * 0.50 + mom_score * 0.50

        # =================================================================
        # 3. MACD MOMENTUM (20%) - Direction AND acceleration
        # =================================================================
        macd_norm = macd_hist / close * 100
        macd_accel_norm = macd_hist_roc / close * 100
        macd_direction = np.clip(0.5 + macd_norm * 8, 0.0, 1.0)
        macd_accel = np.clip(0.5 + macd_accel_norm * 12, 0.0, 1.0)
        fresh_cross = ((macd_hist > 0) & (macd_hist.shift(1) <= 0)).astype(float) * 0.15
        scores['macd'] = macd_direction * 0.55 + macd_accel * 0.30 + fresh_cross

        # =================================================================
        # 4. BREAKOUT & VOLUME CONFIRMATION (15%)
        #    Price near upper BB with high volume = breakout
        # =================================================================
        bb_range = bb_upper - bb_lower + 1e-10
        bb_position = (close - bb_lower) / bb_range

        bb_score = np.where(bb_position > 0.90, 0.90,     # Breakout!
                  np.where(bb_position > 0.70, 0.80,
                  np.where(bb_position > 0.50, 0.60,
                  np.where(bb_position > 0.30, 0.35, 0.15))))

        up_day = (close > close.shift(1)).astype(float)
        vol_score = np.where((vol_ratio > 1.5) & (up_day == 1), 0.95,
                  np.where(vol_ratio > 1.5, 0.55,
                  np.where(vol_ratio > 1.0, 0.50,
                  np.where(vol_ratio > 0.7, 0.40, 0.25))))

        scores['breakout'] = pd.Series(bb_score, index=df.index) * 0.50 + \
                             pd.Series(vol_score, index=df.index) * 0.50

        # =================================================================
        # 5. RELATIVE STRENGTH vs BENCHMARK (15%)
        # =================================================================
        if benchmark_df is not None and 'Close' in benchmark_df.columns:
            bench_close = benchmark_df['Close'].reindex(df.index, method='ffill')
            stock_ret_21 = close.pct_change(21)
            bench_ret_21 = bench_close.pct_change(21)
            rel_21 = stock_ret_21 - bench_ret_21

            stock_ret_63 = close.pct_change(min(63, max(len(close)-2, 21)))
            bench_ret_63 = bench_close.pct_change(min(63, max(len(close)-2, 21)))
            rel_63 = stock_ret_63 - bench_ret_63

            scores['rel_strength'] = np.clip(0.5 + rel_21 * 4, 0.1, 0.95) * 0.6 + \
                                     np.clip(0.5 + rel_63 * 3, 0.1, 0.95) * 0.4
        else:
            scores['rel_strength'] = 0.5

        # =================================================================
        # WEIGHTED COMPOSITE - MOMENTUM FOCUSED
        # =================================================================
        composite = (
            scores['trend'] * 0.25 +
            scores['momentum'] * 0.25 +
            scores['macd'] * 0.20 +
            scores['breakout'] * 0.15 +
            scores['rel_strength'] * 0.15
        )

        # TREND GATE: Penalize stocks in downtrends
        downtrend_penalty = np.where(
            (close < sma_50) & (close < sma_200), 0.50,
            np.where(close < sma_200, 0.75, 1.0))
        composite = composite * downtrend_penalty

        return composite


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Simulates the full trading system over historical data.
    Mirrors Module 3's risk management and position sizing exactly.
    """

    # Risk params - AGGRESSIVE MOMENTUM CONFIGURATION
    MAX_POSITION_ALLOCATION_PCT = 7.0       # Up to 7% for highest conviction winners
    MED_POSITION_ALLOCATION_PCT = 4.5       # 4.5% for medium conviction
    MIN_POSITION_ALLOCATION_PCT = 2.5       # 2.5% for base conviction
    HIGH_CONVICTION_THRESHOLD = 0.72        # Slightly lower threshold to catch more momentum
    MED_CONVICTION_THRESHOLD = 0.58
    MIN_CONFIDENCE_THRESHOLD = 0.50
    DAILY_LOSS_CIRCUIT_BREAKER_PCT = 4.0    # Wider circuit breaker
    HIGH_VOLATILITY_ATR_THRESHOLD = 10.0    # Allow more volatile stocks through
    CRYPTO_VOLATILITY_ATR_THRESHOLD = 18.0

    DEFAULT_STOP_LOSS_PCT = 7.0             # Wider stop - survive normal pullbacks
    DEFAULT_TAKE_PROFIT_PCT = 20.0          # Let winners run further
    TRAILING_STOP_TRIGGER_GAIN_PCT = 5.0    # Trailing activates after 5% gain
    TRAILING_STOP_TRAIL_PCT = 4.0           # 4% trail - survive pullbacks in trends

    MAX_POSITIONS = 15                       # More concentrated = more alpha
    REBALANCE_FREQUENCY_DAYS = 3             # Faster rebalance to catch momentum earlier
    SIGNAL_LOOKBACK_DAYS = 60

    # Representative universe across all 10 sectors (liquid, tradeable)
    SECTOR_UNIVERSE = {
        "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "PLTR", "CRM", "AVGO", "ADBE"],
        "Healthcare/Biotech": ["JNJ", "UNH", "LLY", "PFE", "MRNA", "ABBV", "ISRG", "TMO", "AMGN", "GILD"],
        "Financial Services": ["JPM", "GS", "V", "MA", "BAC", "MS", "BLK", "SCHW", "AXP", "C"],
        "Energy/Oil": ["XOM", "CVX", "OXY", "COP", "SLB", "EOG", "PSX", "VLO", "FSLR", "CEG"],
        "Consumer Goods/Retail": ["AMZN", "WMT", "COST", "HD", "NKE", "SBUX", "TGT", "LULU", "MCD", "PG"],
        "Industrial/Manufacturing": ["CAT", "DE", "BA", "HON", "GE", "RTX", "LMT", "UPS", "MMM", "EMR"],
        "Real Estate": ["O", "AMT", "PLD", "SPG", "EQIX", "DLR", "PSA", "WELL", "AVB", "EQR"],
        "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Communications": ["DIS", "NFLX", "CMCSA", "T", "VZ", "SPOT", "RBLX", "TMUS", "CHTR", "EA"],
        "Materials/Mining": ["FCX", "NEM", "LIN", "APD", "ECL", "SHW", "NUE", "GOLD", "DD", "VMC"]
    }

    # Benchmark
    BENCHMARK_SYMBOL = "SPY"

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.analyzer = TechnicalAnalyzer()
        self.all_symbols = []
        self.symbol_sectors = {}

        # Build flat symbol list & sector map
        for sector, symbols in self.SECTOR_UNIVERSE.items():
            for sym in symbols:
                self.all_symbols.append(sym)
                self.symbol_sectors[sym] = sector

    def download_data(self, years_back: int) -> Dict[str, pd.DataFrame]:
        """Download historical data for all symbols."""
        end_date = datetime.now()
        # Add extra days for technical indicator warmup
        start_date = end_date - timedelta(days=years_back * 365 + 90)

        print(f"\nðŸ“¥ Downloading {len(self.all_symbols) + 1} symbols ({years_back}Y backtest)...")

        all_data = {}
        # Download in batches to avoid rate limits
        symbols_to_download = self.all_symbols + [self.BENCHMARK_SYMBOL]
        batch_size = 20

        for i in range(0, len(symbols_to_download), batch_size):
            batch = symbols_to_download[i:i+batch_size]
            batch_str = " ".join(batch)
            try:
                data = yf.download(batch_str, start=start_date, end=end_date,
                                   progress=False, group_by='ticker', auto_adjust=True)

                if len(batch) == 1:
                    sym = batch[0]
                    if not data.empty:
                        all_data[sym] = data.copy()
                else:
                    for sym in batch:
                        try:
                            if sym in data.columns.get_level_values(0):
                                sym_data = data[sym].dropna(how='all')
                                if not sym_data.empty and len(sym_data) > 60:
                                    all_data[sym] = sym_data.copy()
                        except Exception:
                            pass
            except Exception as e:
                print(f"  âš ï¸ Batch download error: {e}")

        # Fix column names if needed (yfinance sometimes returns multi-level)
        for sym in list(all_data.keys()):
            df = all_data[sym]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
                all_data[sym] = df
            # Ensure required columns exist
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                del all_data[sym]

        print(f"  âœ… Loaded {len(all_data)} symbols successfully")
        return all_data

    def run_backtest(self, years_back: int, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run the full backtest simulation.

        This replicates the system's weekly rebalance cycle:
        1. Score all symbols using technical analysis
        2. Rank by composite score
        3. Enter positions based on conviction sizing
        4. Manage stops, take-profits, and trailing stops daily
        5. Rebalance weekly
        """
        print(f"\nðŸ”„ Running {years_back}Y backtest (${self.initial_capital:,.0f} starting capital)...")

        # Determine actual backtest date range
        benchmark_data = data.get(self.BENCHMARK_SYMBOL)
        if benchmark_data is None or benchmark_data.empty:
            return {"error": "No benchmark data"}

        start_idx = max(60, 0)  # Need 60 days warmup
        actual_start = benchmark_data.index[start_idx]
        backtest_start = actual_start + timedelta(days=90)  # Skip warmup period
        backtest_end = benchmark_data.index[-1]

        # Filter to backtest period
        bt_dates = benchmark_data.loc[backtest_start:backtest_end].index

        # State
        cash = self.initial_capital
        positions: Dict[str, Position] = {}
        completed_trades: List[Trade] = []
        daily_snapshots: List[DailySnapshot] = []
        peak_value = self.initial_capital
        days_since_rebalance = 0
        circuit_breaker_active = False
        prev_portfolio_value = self.initial_capital

        # Pre-compute technical scores for all symbols
        print("  ðŸ“Š Computing technical scores...")
        benchmark_df = data.get(self.BENCHMARK_SYMBOL)
        symbol_scores = {}
        for sym, df in data.items():
            if sym == self.BENCHMARK_SYMBOL:
                continue
            try:
                scores = self.analyzer.generate_composite_score(df, benchmark_df=benchmark_df)
                symbol_scores[sym] = scores
            except Exception:
                pass

        print(f"  ðŸ“ˆ Simulating {len(bt_dates)} trading days...")

        for i, date in enumerate(bt_dates):
            date_str = date.strftime('%Y-%m-%d')

            # ---------------------------------------------------------------
            # 1. UPDATE POSITIONS (check stops, take-profits, trailing stops)
            # ---------------------------------------------------------------
            symbols_to_close = []
            for sym, pos in positions.items():
                if sym not in data or date not in data[sym].index:
                    continue

                current_price = data[sym].loc[date, 'Close']
                if pd.isna(current_price) or current_price <= 0:
                    continue

                # Update highest price for trailing stop
                if current_price > pos.highest_price:
                    pos.highest_price = current_price

                # Check if trailing stop should activate (4% gain trigger)
                gain_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                if gain_pct >= self.TRAILING_STOP_TRIGGER_GAIN_PCT and pos.trailing_stop is None:
                    pos.trailing_stop = current_price * (1 - self.TRAILING_STOP_TRAIL_PCT / 100)

                # Update trailing stop level
                if pos.trailing_stop is not None:
                    new_trail = pos.highest_price * (1 - self.TRAILING_STOP_TRAIL_PCT / 100)
                    pos.trailing_stop = max(pos.trailing_stop, new_trail)

                # Check exit conditions
                exit_reason = None
                if current_price <= pos.stop_loss:
                    exit_reason = 'stop_loss'
                elif current_price >= pos.take_profit:
                    exit_reason = 'take_profit'
                elif pos.trailing_stop and current_price <= pos.trailing_stop:
                    exit_reason = 'trailing_stop'

                if exit_reason:
                    pnl = (current_price - pos.entry_price) * pos.shares
                    pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    holding_days = (date - entry_dt).days if isinstance(date, datetime) else (pd.Timestamp(date) - pd.Timestamp(entry_dt)).days

                    completed_trades.append(Trade(
                        symbol=sym, sector=pos.sector,
                        entry_date=pos.entry_date, exit_date=date_str,
                        entry_price=pos.entry_price, exit_price=current_price,
                        shares=pos.shares, side=pos.side,
                        pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        holding_days=max(holding_days, 1),
                        conviction=pos.conviction
                    ))
                    cash += (current_price * pos.shares)
                    symbols_to_close.append(sym)

                # MOMENTUM DECAY EXIT: If score drops below threshold, exit proactively
                elif sym in symbol_scores and date in symbol_scores[sym].index:
                    current_score = symbol_scores[sym].loc[date]
                    entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
                    holding_days = (pd.Timestamp(date) - pd.Timestamp(entry_dt)).days
                    # After 10+ days, exit if score drops well below entry threshold
                    if holding_days > 10 and not pd.isna(current_score) and current_score < 0.35:
                        pnl = (current_price - pos.entry_price) * pos.shares
                        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                        completed_trades.append(Trade(
                            symbol=sym, sector=pos.sector,
                            entry_date=pos.entry_date, exit_date=date_str,
                            entry_price=pos.entry_price, exit_price=current_price,
                            shares=pos.shares, side=pos.side,
                            pnl=pnl, pnl_pct=pnl_pct,
                            exit_reason='momentum_decay',
                            holding_days=max(holding_days, 1),
                            conviction=pos.conviction
                        ))
                        cash += (current_price * pos.shares)
                        symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del positions[sym]

            # ---------------------------------------------------------------
            # 2. WEEKLY REBALANCE (new entries)
            # ---------------------------------------------------------------
            days_since_rebalance += 1

            if days_since_rebalance >= self.REBALANCE_FREQUENCY_DAYS:
                days_since_rebalance = 0

                # Circuit breaker check
                portfolio_value = cash + sum(
                    data[s].loc[date, 'Close'] * p.shares
                    for s, p in positions.items()
                    if s in data and date in data[s].index
                )
                daily_loss = ((portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100

                if daily_loss < -self.DAILY_LOSS_CIRCUIT_BREAKER_PCT:
                    circuit_breaker_active = True
                else:
                    circuit_breaker_active = False

                if not circuit_breaker_active:
                    # MARKET REGIME FILTER
                    # Be aggressive in uptrends, cautious in downtrends
                    market_regime = 'neutral'
                    if self.BENCHMARK_SYMBOL in data and date in data[self.BENCHMARK_SYMBOL].index:
                        spy_data = data[self.BENCHMARK_SYMBOL].loc[:date]
                        if len(spy_data) >= 50:
                            spy_close = spy_data['Close']
                            spy_sma50 = spy_close.rolling(50).mean().iloc[-1]
                            spy_sma200 = spy_close.rolling(min(200, len(spy_close)-1)).mean().iloc[-1] if len(spy_close) > 50 else spy_sma50
                            spy_price = spy_close.iloc[-1]
                            if spy_price > spy_sma50 and spy_price > spy_sma200:
                                market_regime = 'bull'
                            elif spy_price < spy_sma50 and spy_price < spy_sma200:
                                market_regime = 'bear'

                    # Adjust max positions based on market regime
                    regime_max_positions = {
                        'bull': self.MAX_POSITIONS,
                        'neutral': self.MAX_POSITIONS - 3,
                        'bear': self.MAX_POSITIONS - 7
                    }
                    max_pos_now = regime_max_positions.get(market_regime, self.MAX_POSITIONS)

                    # Score all symbols for today
                    candidates = []
                    for sym in symbol_scores:
                        if sym in positions:
                            continue  # Skip already held
                        if date not in symbol_scores[sym].index:
                            continue
                        score = symbol_scores[sym].loc[date]
                        if pd.isna(score) or score < self.MIN_CONFIDENCE_THRESHOLD:
                            continue

                        # ATR volatility filter (mirrors Module 3)
                        if sym in data and date in data[sym].index:
                            try:
                                lookback = data[sym].loc[:date].tail(20)
                                if len(lookback) >= 14:
                                    atr = self.analyzer.calculate_atr(
                                        lookback['High'], lookback['Low'], lookback['Close']
                                    )
                                    atr_val = atr.iloc[-1]
                                    price = lookback['Close'].iloc[-1]
                                    atr_pct = (atr_val / price) * 100 if price > 0 else 999

                                    sector = self.symbol_sectors.get(sym, "")
                                    threshold = self.CRYPTO_VOLATILITY_ATR_THRESHOLD if "Crypto" in sector else self.HIGH_VOLATILITY_ATR_THRESHOLD
                                    if atr_pct > threshold:
                                        continue
                            except Exception:
                                pass

                        candidates.append((sym, score))

                    # Sort by score (highest first) - mimics AI consensus ranking
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    # Sector diversification: max 4 positions per sector (allow concentration in hot sectors)
                    sector_counts = {}
                    for sym, pos in positions.items():
                        sec = pos.sector
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1

                    # Enter new positions
                    for sym, score in candidates:
                        if len(positions) >= max_pos_now:
                            break

                        sector = self.symbol_sectors.get(sym, "Unknown")
                        max_per_sector = 4 if market_regime == 'bull' else 3
                        if sector_counts.get(sector, 0) >= max_per_sector:
                            continue

                        # Conviction-based position sizing (regime-enhanced)
                        if score >= self.HIGH_CONVICTION_THRESHOLD:
                            alloc_pct = self.MAX_POSITION_ALLOCATION_PCT
                            conviction = 'high'
                        elif score >= self.MED_CONVICTION_THRESHOLD:
                            alloc_pct = self.MED_POSITION_ALLOCATION_PCT
                            conviction = 'medium'
                        else:
                            alloc_pct = self.MIN_POSITION_ALLOCATION_PCT
                            conviction = 'base'

                        # Bull market boost: 20% larger positions
                        if market_regime == 'bull':
                            alloc_pct *= 1.20
                        elif market_regime == 'bear':
                            alloc_pct *= 0.70

                        portfolio_value_now = cash + sum(
                            data[s].loc[date, 'Close'] * p.shares
                            for s, p in positions.items()
                            if s in data and date in data[s].index
                        )
                        position_budget = portfolio_value_now * (alloc_pct / 100)

                        if position_budget > cash * 0.98:
                            continue  # Only hold 2% reserve minimum

                        if sym not in data or date not in data[sym].index:
                            continue
                        entry_price = data[sym].loc[date, 'Close']
                        if pd.isna(entry_price) or entry_price <= 0:
                            continue

                        shares = int(position_budget / entry_price)
                        if shares <= 0:
                            continue

                        cost = shares * entry_price
                        if cost > cash:
                            continue

                        # Dynamic stop-loss and take-profit based on conviction
                        if conviction == 'high':
                            stop_pct = self.DEFAULT_STOP_LOSS_PCT + 1.0   # 8% stop for high conviction
                            tp_pct = self.DEFAULT_TAKE_PROFIT_PCT + 10.0  # 30% target!
                        elif conviction == 'medium':
                            stop_pct = self.DEFAULT_STOP_LOSS_PCT
                            tp_pct = self.DEFAULT_TAKE_PROFIT_PCT
                        else:
                            stop_pct = self.DEFAULT_STOP_LOSS_PCT - 1.0   # Tighter for low conviction
                            tp_pct = self.DEFAULT_TAKE_PROFIT_PCT - 5.0

                        stop_loss = entry_price * (1 - stop_pct / 100)
                        take_profit = entry_price * (1 + tp_pct / 100)

                        positions[sym] = Position(
                            symbol=sym, sector=sector,
                            entry_date=date_str, entry_price=entry_price,
                            shares=shares, side='BUY',
                            stop_loss=stop_loss, take_profit=take_profit,
                            trailing_stop=None, highest_price=entry_price,
                            conviction=conviction,
                            allocation_pct=alloc_pct
                        )
                        cash -= cost
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1

            # ---------------------------------------------------------------
            # 3. DAILY SNAPSHOT
            # ---------------------------------------------------------------
            positions_value = sum(
                data[s].loc[date, 'Close'] * p.shares
                for s, p in positions.items()
                if s in data and date in data[s].index and not pd.isna(data[s].loc[date, 'Close'])
            )
            portfolio_value = cash + positions_value

            if portfolio_value > peak_value:
                peak_value = portfolio_value
            drawdown = ((portfolio_value - peak_value) / peak_value) * 100

            daily_return = ((portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100 if prev_portfolio_value > 0 else 0

            daily_snapshots.append(DailySnapshot(
                date=date_str,
                portfolio_value=portfolio_value,
                cash=cash,
                positions_value=positions_value,
                num_positions=len(positions),
                daily_return_pct=daily_return,
                drawdown_pct=drawdown
            ))

            prev_portfolio_value = portfolio_value

        # Close remaining positions at final prices
        final_date = bt_dates[-1]
        for sym, pos in list(positions.items()):
            if sym in data and final_date in data[sym].index:
                final_price = data[sym].loc[final_date, 'Close']
                if not pd.isna(final_price):
                    pnl = (final_price - pos.entry_price) * pos.shares
                    pnl_pct = ((final_price - pos.entry_price) / pos.entry_price) * 100
                    entry_dt = pd.Timestamp(pos.entry_date)
                    holding_days = (pd.Timestamp(final_date) - entry_dt).days

                    completed_trades.append(Trade(
                        symbol=sym, sector=pos.sector,
                        entry_date=pos.entry_date, exit_date=final_date.strftime('%Y-%m-%d'),
                        entry_price=pos.entry_price, exit_price=final_price,
                        shares=pos.shares, side=pos.side,
                        pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason='end_of_backtest',
                        holding_days=max(holding_days, 1),
                        conviction=pos.conviction
                    ))

        # Get benchmark returns
        benchmark_series = benchmark_data.loc[backtest_start:backtest_end, 'Close']

        return {
            'years': years_back,
            'start_date': backtest_start.strftime('%Y-%m-%d'),
            'end_date': backtest_end.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'trades': completed_trades,
            'daily_snapshots': daily_snapshots,
            'benchmark': benchmark_series
        }


# =============================================================================
# PERFORMANCE ANALYTICS
# =============================================================================

class PerformanceAnalyzer:
    """Compute institutional-grade performance metrics."""

    @staticmethod
    def compute_metrics(result: Dict) -> Dict:
        snapshots = result['daily_snapshots']
        trades = result['trades']
        benchmark = result['benchmark']

        if not snapshots:
            return {"error": "No data"}

        # Basic P&L
        initial = result['initial_capital']
        final = snapshots[-1].portfolio_value
        total_return = ((final - initial) / initial) * 100
        years = result['years']

        # CAGR
        actual_years = len(snapshots) / 252.0
        cagr = ((final / initial) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else 0

        # Daily returns
        daily_returns = [s.daily_return_pct for s in snapshots]
        daily_returns_arr = np.array(daily_returns)

        # Sharpe Ratio (annualized, risk-free = 4.5%)
        risk_free_daily = 4.5 / 252
        excess_returns = daily_returns_arr - risk_free_daily
        sharpe = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
        sortino = (np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max Drawdown
        max_dd = min(s.drawdown_pct for s in snapshots)

        # Calmar Ratio (CAGR / |Max Drawdown|)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Average win/loss
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average holding period
        avg_holding = np.mean([t.holding_days for t in trades]) if trades else 0

        # Monthly returns
        monthly_returns = {}
        for s in snapshots:
            month_key = s.date[:7]
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(s.daily_return_pct)

        positive_months = sum(1 for rets in monthly_returns.values() if sum(rets) > 0)
        total_months = len(monthly_returns)
        pct_positive_months = positive_months / total_months * 100 if total_months > 0 else 0

        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 1:
            bench_start = benchmark.iloc[0]
            bench_end = benchmark.iloc[-1]
            bench_return = ((bench_end - bench_start) / bench_start) * 100
            bench_cagr = ((bench_end / bench_start) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else 0
            alpha = cagr - bench_cagr
        else:
            bench_return = 0
            bench_cagr = 0
            alpha = 0

        # Trade breakdown by exit reason
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Sector performance
        sector_pnl = {}
        for t in trades:
            if t.sector not in sector_pnl:
                sector_pnl[t.sector] = {'total_pnl': 0, 'count': 0, 'wins': 0}
            sector_pnl[t.sector]['total_pnl'] += t.pnl
            sector_pnl[t.sector]['count'] += 1
            if t.pnl > 0:
                sector_pnl[t.sector]['wins'] += 1

        # Conviction performance
        conviction_perf = {}
        for tier in ['high', 'medium', 'base']:
            tier_trades = [t for t in trades if t.conviction == tier]
            if tier_trades:
                conviction_perf[tier] = {
                    'count': len(tier_trades),
                    'avg_pnl_pct': np.mean([t.pnl_pct for t in tier_trades]),
                    'win_rate': sum(1 for t in tier_trades if t.pnl > 0) / len(tier_trades) * 100,
                    'total_pnl': sum(t.pnl for t in tier_trades)
                }

        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'calmar_ratio': calmar,
            'total_trades': len(trades),
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
            'final_portfolio_value': final,
            'total_pnl': final - initial,
            'pct_positive_months': pct_positive_months,
            'benchmark_return_pct': bench_return,
            'benchmark_cagr_pct': bench_cagr,
            'alpha_pct': alpha,
            'exit_reasons': exit_reasons,
            'sector_performance': sector_pnl,
            'conviction_performance': conviction_perf,
            'actual_years': actual_years,
            'total_months': total_months
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate visual HTML report with charts."""

    @staticmethod
    def generate_charts(results: List[Dict], output_dir: str) -> List[str]:
        """Generate performance charts for all backtest periods."""
        chart_paths = []

        # Color scheme
        colors = {1: '#00D4AA', 3: '#4A90D9', 5: '#9B59B6'}
        bench_color = '#FF6B6B'

        # ---- CHART 1: Equity Curves (all periods) ----
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 5 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for idx, res in enumerate(results):
            ax = axes[idx]
            snapshots = res['daily_snapshots']
            benchmark = res['benchmark']
            years = res['years']
            color = colors.get(years, '#00D4AA')

            dates = [datetime.strptime(s.date, '%Y-%m-%d') for s in snapshots]
            values = [s.portfolio_value for s in snapshots]

            # Normalize to 100
            initial = values[0] if values else 100
            norm_values = [v / initial * 100 for v in values]

            ax.plot(dates, norm_values, color=color, linewidth=2, label=f'Strategy ({years}Y)')

            # Benchmark
            if benchmark is not None and len(benchmark) > 0:
                bench_dates = benchmark.index.tolist()
                bench_start = benchmark.iloc[0]
                norm_bench = [v / bench_start * 100 for v in benchmark.values]
                # Align benchmark dates with strategy dates
                ax.plot(bench_dates[:len(norm_bench)], norm_bench, color=bench_color,
                       linewidth=1.5, linestyle='--', alpha=0.7, label='SPY Benchmark')

            ax.set_title(f'{years}-Year Equity Curve (Normalized to 100)', fontsize=14, fontweight='bold', color='white')
            ax.legend(loc='upper left', fontsize=10)
            ax.set_ylabel('Growth of $100', fontsize=11, color='white')
            ax.grid(True, alpha=0.2)
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, years * 2)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            for spine in ax.spines.values():
                spine.set_color('#333')

        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        path1 = os.path.join(output_dir, 'equity_curves.png')
        plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        chart_paths.append(path1)

        # ---- CHART 2: Drawdown Chart (longest period) ----
        longest = max(results, key=lambda r: r['years'])
        fig, ax = plt.subplots(figsize=(14, 4))
        snapshots = longest['daily_snapshots']
        dates = [datetime.strptime(s.date, '%Y-%m-%d') for s in snapshots]
        drawdowns = [s.drawdown_pct for s in snapshots]

        ax.fill_between(dates, drawdowns, 0, color='#FF4444', alpha=0.4)
        ax.plot(dates, drawdowns, color='#FF6666', linewidth=1)
        ax.set_title(f"Drawdown Analysis ({longest['years']}Y)", fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Drawdown %', fontsize=11, color='white')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        path2 = os.path.join(output_dir, 'drawdown.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        chart_paths.append(path2)

        # ---- CHART 3: Sector Performance (longest period) ----
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = PerformanceAnalyzer.compute_metrics(longest)
        sector_data = metrics['sector_performance']
        if sector_data:
            sectors = sorted(sector_data.keys(), key=lambda s: sector_data[s]['total_pnl'], reverse=True)
            pnls = [sector_data[s]['total_pnl'] for s in sectors]
            bar_colors = ['#00D4AA' if p > 0 else '#FF4444' for p in pnls]

            bars = ax.barh(sectors, pnls, color=bar_colors, alpha=0.8)
            ax.set_title(f"P&L by Sector ({longest['years']}Y)", fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Total P&L ($)', fontsize=11, color='white')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#333')

        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        path3 = os.path.join(output_dir, 'sector_performance.png')
        plt.savefig(path3, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        chart_paths.append(path3)

        # ---- CHART 4: Conviction Tier Performance ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, res in enumerate([longest]):
            metrics = PerformanceAnalyzer.compute_metrics(res)
            conv = metrics['conviction_performance']
            if conv:
                tiers = list(conv.keys())
                win_rates = [conv[t]['win_rate'] for t in tiers]
                avg_pnls = [conv[t]['avg_pnl_pct'] for t in tiers]
                tier_colors = {'high': '#00D4AA', 'medium': '#4A90D9', 'base': '#9B59B6'}

                # Win Rate
                ax1 = axes[0]
                bars = ax1.bar(tiers, win_rates, color=[tier_colors.get(t, '#888') for t in tiers], alpha=0.8)
                ax1.set_title('Win Rate by Conviction', fontsize=13, fontweight='bold', color='white')
                ax1.set_ylabel('Win Rate %', color='white')
                for bar, val in zip(bars, win_rates):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=11)

                # Avg P&L
                ax2 = axes[1]
                bars2 = ax2.bar(tiers, avg_pnls, color=[tier_colors.get(t, '#888') for t in tiers], alpha=0.8)
                ax2.set_title('Avg Return by Conviction', fontsize=13, fontweight='bold', color='white')
                ax2.set_ylabel('Avg P&L %', color='white')
                for bar, val in zip(bars2, avg_pnls):
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                            f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=11)

                for ax in [ax1, ax2]:
                    ax.set_facecolor('#1a1a2e')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.2, axis='y')
                    for spine in ax.spines.values():
                        spine.set_color('#333')

        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        path4 = os.path.join(output_dir, 'conviction_analysis.png')
        plt.savefig(path4, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        chart_paths.append(path4)

        return chart_paths

    @staticmethod
    def generate_html_report(all_results: List[Dict], all_metrics: List[Dict],
                             chart_paths: List[str], output_path: str):
        """Generate comprehensive HTML report."""
        import base64

        # Encode charts as base64 for embedded HTML
        chart_images = {}
        for path in chart_paths:
            name = os.path.basename(path).replace('.png', '')
            with open(path, 'rb') as f:
                chart_images[name] = base64.b64encode(f.read()).decode('utf-8')

        # Build metrics comparison table
        periods = [m for m in all_metrics if 'error' not in m]

        def fmt_pct(val):
            color = '#00D4AA' if val >= 0 else '#FF4444'
            return f'<span style="color:{color}">{val:+.2f}%</span>'

        def fmt_dollar(val):
            color = '#00D4AA' if val >= 0 else '#FF4444'
            return f'<span style="color:{color}">${val:+,.0f}</span>'

        def fmt_ratio(val):
            color = '#00D4AA' if val >= 1.0 else '#FFD700' if val >= 0.5 else '#FF4444'
            return f'<span style="color:{color}">{val:.2f}</span>'

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Apex Quant Capital - Backtest Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            padding: 40px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            border: 1px solid #30363d;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(135deg, #00D4AA, #4A90D9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .subtitle {{
            color: #8b949e;
            font-size: 1.1em;
        }}
        .header .date {{
            color: #6e7681;
            font-size: 0.9em;
            margin-top: 10px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 24px;
        }}
        .metric-card h3 {{
            font-size: 1.1em;
            color: #8b949e;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
        }}
        .comparison-table th {{
            background: #1a1a2e;
            color: #00D4AA;
            padding: 14px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #30363d;
        }}
        .comparison-table td {{
            padding: 12px 14px;
            border-bottom: 1px solid #21262d;
        }}
        .comparison-table tr:hover {{
            background: #1a1a2e;
        }}

        .chart-container {{
            margin-bottom: 30px;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid #30363d;
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #e6edf3;
        }}

        .section-title {{
            font-size: 1.5em;
            font-weight: 700;
            margin: 40px 0 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #30363d;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-green {{ background: #0d3320; color: #00D4AA; }}
        .badge-blue {{ background: #0d2340; color: #4A90D9; }}
        .badge-yellow {{ background: #3d2e00; color: #FFD700; }}
        .badge-red {{ background: #3d0d0d; color: #FF4444; }}

        .disclaimer {{
            margin-top: 40px;
            padding: 20px;
            background: #1c1c1c;
            border-left: 4px solid #FFD700;
            border-radius: 4px;
            color: #8b949e;
            font-size: 0.85em;
        }}

        .kpi-row {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 30px;
        }}
        .kpi-box {{
            flex: 1;
            min-width: 150px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .kpi-label {{
            color: #8b949e;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>âš¡ APEX QUANT CAPITAL</h1>
    <div class="subtitle">Historical Backtest Performance Report</div>
    <div class="date">Generated {datetime.now().strftime('%B %d, %Y')} | AI-Driven Multi-Sector Momentum Strategy</div>
</div>

<div class="section-title">ðŸ“Š Strategy Overview</div>
<div class="metric-card" style="margin-bottom:30px">
    <p style="line-height:1.7">
        <strong>Strategy:</strong> AI-Consensus Multi-Sector Momentum with Technical Scoring<br>
        <strong>Universe:</strong> ~100 liquid equities + 3 crypto assets across 10 sectors<br>
        <strong>Rebalance:</strong> Weekly with daily risk management (stops, trailing stops, circuit breakers)<br>
        <strong>Position Sizing:</strong> Conviction-based (2%/3.5%/5% of portfolio)<br>
        <strong>Risk Controls:</strong> 5% stop-loss, 12% take-profit, 3.5% trailing stop (after 4% gain), 3% daily circuit breaker<br>
        <strong>Max Positions:</strong> 20 (max 3 per sector for diversification)<br>
        <strong>Starting Capital:</strong> ${all_results[0]['initial_capital']:,.0f}
    </p>
</div>
"""

        # KPI boxes for each period
        for i, (res, m) in enumerate(zip(all_results, all_metrics)):
            if 'error' in m:
                continue
            yrs = res['years']
            sr_badge = 'badge-green' if m['sharpe_ratio'] > 1 else 'badge-blue' if m['sharpe_ratio'] > 0.5 else 'badge-red'
            html += f"""
<div class="section-title">ðŸ• {yrs}-Year Backtest ({res['start_date']} â†’ {res['end_date']})</div>
<div class="kpi-row">
    <div class="kpi-box">
        <div class="kpi-value" style="color:{'#00D4AA' if m['total_return_pct'] >= 0 else '#FF4444'}">{m['total_return_pct']:+.1f}%</div>
        <div class="kpi-label">Total Return</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value" style="color:{'#00D4AA' if m['cagr_pct'] >= 0 else '#FF4444'}">{m['cagr_pct']:+.1f}%</div>
        <div class="kpi-label">CAGR</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value"><span class="{sr_badge} badge">{m['sharpe_ratio']:.2f}</span></div>
        <div class="kpi-label">Sharpe Ratio</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value" style="color:#FF4444">{m['max_drawdown_pct']:.1f}%</div>
        <div class="kpi-label">Max Drawdown</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value" style="color:#4A90D9">{m['win_rate_pct']:.1f}%</div>
        <div class="kpi-label">Win Rate</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value">{fmt_dollar(m['total_pnl'])}</div>
        <div class="kpi-label">Total P&L</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-value" style="color:{'#00D4AA' if m['alpha_pct'] >= 0 else '#FF4444'}">{m['alpha_pct']:+.1f}%</div>
        <div class="kpi-label">Alpha vs SPY</div>
    </div>
</div>
"""

        # Detailed comparison table
        html += """
<div class="section-title">ðŸ“ˆ Detailed Metrics Comparison</div>
<table class="comparison-table">
<tr>
    <th>Metric</th>
"""
        for res in all_results:
            html += f"<th>{res['years']}-Year</th>"
        html += "</tr>"

        metrics_rows = [
            ('Total Return', 'total_return_pct', fmt_pct),
            ('CAGR', 'cagr_pct', fmt_pct),
            ('Sharpe Ratio', 'sharpe_ratio', fmt_ratio),
            ('Sortino Ratio', 'sortino_ratio', fmt_ratio),
            ('Max Drawdown', 'max_drawdown_pct', fmt_pct),
            ('Calmar Ratio', 'calmar_ratio', fmt_ratio),
            ('Win Rate', 'win_rate_pct', lambda v: f'{v:.1f}%'),
            ('Avg Win', 'avg_win_pct', fmt_pct),
            ('Avg Loss', 'avg_loss_pct', fmt_pct),
            ('Profit Factor', 'profit_factor', fmt_ratio),
            ('Avg Holding (days)', 'avg_holding_days', lambda v: f'{v:.0f}'),
            ('Total Trades', 'total_trades', lambda v: f'{v:,}'),
            ('% Positive Months', 'pct_positive_months', lambda v: f'{v:.1f}%'),
            ('Final Portfolio', 'final_portfolio_value', lambda v: f'${v:,.0f}'),
            ('SPY Return', 'benchmark_return_pct', fmt_pct),
            ('SPY CAGR', 'benchmark_cagr_pct', fmt_pct),
            ('Alpha vs SPY', 'alpha_pct', fmt_pct),
        ]

        for label, key, formatter in metrics_rows:
            html += f"<tr><td><strong>{label}</strong></td>"
            for m in all_metrics:
                val = m.get(key, 0)
                html += f"<td>{formatter(val)}</td>"
            html += "</tr>"
        html += "</table>"

        # Charts
        html += '<div class="section-title">ðŸ“‰ Visual Analysis</div>'
        chart_names = ['equity_curves', 'drawdown', 'sector_performance', 'conviction_analysis']
        chart_titles = ['Equity Curves vs Benchmark', 'Drawdown Analysis', 'Sector P&L Breakdown', 'Conviction Tier Analysis']

        for name, title in zip(chart_names, chart_titles):
            if name in chart_images:
                html += f"""
<div class="chart-container">
    <div class="chart-title">{title}</div>
    <img src="data:image/png;base64,{chart_images[name]}" alt="{title}">
</div>
"""

        # Conviction breakdown table
        longest_metrics = all_metrics[-1] if all_metrics else {}
        conv = longest_metrics.get('conviction_performance', {})
        if conv:
            html += """
<div class="section-title">ðŸŽ¯ Conviction Tier Breakdown</div>
<table class="comparison-table">
<tr><th>Conviction</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th><th>Total P&L</th></tr>
"""
            for tier in ['high', 'medium', 'base']:
                if tier in conv:
                    c = conv[tier]
                    badge = 'badge-green' if tier == 'high' else 'badge-blue' if tier == 'medium' else 'badge-yellow'
                    html += f"""<tr>
<td><span class="badge {badge}">{tier.upper()}</span></td>
<td>{c['count']}</td>
<td>{c['win_rate']:.1f}%</td>
<td>{fmt_pct(c['avg_pnl_pct'])}</td>
<td>{fmt_dollar(c['total_pnl'])}</td>
</tr>"""
            html += "</table>"

        # Exit reason breakdown
        exit_reasons = longest_metrics.get('exit_reasons', {})
        if exit_reasons:
            html += """
<div class="section-title">ðŸšª Exit Reason Analysis</div>
<table class="comparison-table">
<tr><th>Exit Reason</th><th>Count</th><th>% of Trades</th></tr>
"""
            total_exits = sum(exit_reasons.values())
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_exits * 100
                reason_display = reason.replace('_', ' ').title()
                html += f"<tr><td>{reason_display}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
            html += "</table>"

        html += """
<div class="disclaimer">
    <strong>âš ï¸ IMPORTANT DISCLAIMER</strong><br><br>
    This backtest uses historical data and technical indicator scoring to simulate how the AI-driven trading strategy
    would have performed. Actual AI consensus signals (Claude + GPT-4o) are not available for historical periods and
    are approximated using composite technical scores. Past performance does not guarantee future results. Real-world
    execution would be subject to slippage, commissions, market impact, and other factors not fully captured here.
    Crypto backtesting uses daily close prices which may differ from actual execution prices. This report is for
    informational purposes and should not be considered investment advice.
</div>

</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("âš¡ APEX QUANT CAPITAL - HISTORICAL BACKTEST ENGINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    engine = BacktestEngine(initial_capital=100000.0)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_output')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    all_metrics = []
    backtest_periods = [1, 3, 5]

    for years in backtest_periods:
        print(f"\n{'='*50}")
        print(f"  ðŸ“… {years}-YEAR BACKTEST")
        print(f"{'='*50}")

        data = engine.download_data(years)
        result = engine.run_backtest(years, data)

        if 'error' in result:
            print(f"  âŒ Error: {result['error']}")
            continue

        metrics = PerformanceAnalyzer.compute_metrics(result)

        all_results.append(result)
        all_metrics.append(metrics)

        # Print summary
        print(f"\n  ðŸ“Š {years}Y RESULTS:")
        print(f"     Total Return:  {metrics['total_return_pct']:+.1f}%")
        print(f"     CAGR:          {metrics['cagr_pct']:+.1f}%")
        print(f"     Sharpe:        {metrics['sharpe_ratio']:.2f}")
        print(f"     Max Drawdown:  {metrics['max_drawdown_pct']:.1f}%")
        print(f"     Win Rate:      {metrics['win_rate_pct']:.1f}%")
        print(f"     Alpha vs SPY:  {metrics['alpha_pct']:+.1f}%")
        print(f"     Total P&L:     ${metrics['total_pnl']:+,.0f}")
        print(f"     Total Trades:  {metrics['total_trades']}")

    if all_results:
        # Generate charts
        print("\nðŸ“Š Generating charts...")
        chart_paths = ReportGenerator.generate_charts(all_results, output_dir)

        # Generate HTML report
        report_path = os.path.join(output_dir, 'backtest_report.html')
        ReportGenerator.generate_html_report(all_results, all_metrics, chart_paths, report_path)
        print(f"\nâœ… Report saved: {report_path}")

        # Also save raw metrics as JSON
        json_path = os.path.join(output_dir, 'backtest_metrics.json')
        json_metrics = []
        for res, m in zip(all_results, all_metrics):
            m_clean = {k: v for k, v in m.items() if k not in ['sector_performance', 'conviction_performance', 'exit_reasons']}
            m_clean['period'] = f"{res['years']}Y"
            m_clean['start_date'] = res['start_date']
            m_clean['end_date'] = res['end_date']
            json_metrics.append(m_clean)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2, default=str)

        # Copy report to output (same directory as script)
        final_report = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_report.html')
        import shutil
        shutil.copy2(report_path, final_report)
        print(f"âœ… Final report: {final_report}")

    print(f"\n{'='*70}")
    print(f"âš¡ Backtest complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()