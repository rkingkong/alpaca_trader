#!/usr/bin/env python3
"""
Module 0: Market Regime Detector & Sector Rotation Engine
===========================================================
Classifies market regime and provides adaptive parameters for all downstream modules.

Regime Types:
  BULL_STRONG  → Max aggression, wide stops, let winners run
  BULL         → Aggressive, momentum-focused
  SIDEWAYS     → Selective, tighter risk management
  BEAR         → Defensive, smaller positions, more cash
  BEAR_STRONG  → Capital preservation mode

Provides:
  1. Market regime classification via SPY analysis
  2. Sector relative strength rankings (rotation signals)
  3. Adaptive parameter recommendations for all modules
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Shared config
try:
    from config import (
        Config, RegimeContext, save_regime_context,
        SECTOR_ETFS, TOP_FUND_BENCHMARKS, DATA_DIR, print_header,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        Config, RegimeContext, save_regime_context,
        SECTOR_ETFS, TOP_FUND_BENCHMARKS, DATA_DIR, print_header,
    )


class MarketRegime(Enum):
    BULL_STRONG = "bull_strong"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    BEAR_STRONG = "bear_strong"


# Regime → adaptive parameter presets
REGIME_PARAMS = {
    MarketRegime.BULL_STRONG: {
        "position_size_pct": 7.0, "max_positions": 25,
        "stop_loss_pct": 8.0, "take_profit_pct": 25.0,
        "cash_reserve_pct": 2.0, "confidence_threshold": 0.40,
        "risk_appetite": 0.90,
    },
    MarketRegime.BULL: {
        "position_size_pct": 5.0, "max_positions": 20,
        "stop_loss_pct": 7.0, "take_profit_pct": 20.0,
        "cash_reserve_pct": 3.0, "confidence_threshold": 0.45,
        "risk_appetite": 0.75,
    },
    MarketRegime.SIDEWAYS: {
        "position_size_pct": 3.5, "max_positions": 15,
        "stop_loss_pct": 5.0, "take_profit_pct": 12.0,
        "cash_reserve_pct": 10.0, "confidence_threshold": 0.55,
        "risk_appetite": 0.50,
    },
    MarketRegime.BEAR: {
        "position_size_pct": 2.5, "max_positions": 10,
        "stop_loss_pct": 4.0, "take_profit_pct": 8.0,
        "cash_reserve_pct": 20.0, "confidence_threshold": 0.65,
        "risk_appetite": 0.30,
    },
    MarketRegime.BEAR_STRONG: {
        "position_size_pct": 1.5, "max_positions": 5,
        "stop_loss_pct": 3.0, "take_profit_pct": 5.0,
        "cash_reserve_pct": 40.0, "confidence_threshold": 0.75,
        "risk_appetite": 0.10,
    },
}


class RegimeDetector:
    """Detects market regime from SPY price action and sector breadth."""

    def __init__(self):
        cfg = Config()
        self.data_client = StockHistoricalDataClient(
            cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY
        )

    def detect(self) -> RegimeContext:
        """Run full regime detection and return context."""
        print("  🔍 Detecting market regime...")

        spy_data = self._fetch("SPY", 250)
        if spy_data is None or len(spy_data) < 50:
            print("  ⚠️  Insufficient SPY data — defaulting to SIDEWAYS")
            return RegimeContext()

        # Analyze SPY
        trend = self._analyze_trend(spy_data)
        vol = self._analyze_volatility(spy_data)

        # Sector analysis
        sector_rs = self._sector_relative_strength()
        sector_mom = self._sector_momentum()

        # Combine into regime
        score = self._regime_score(trend, vol)
        regime = self._score_to_regime(score)
        params = REGIME_PARAMS.get(regime, REGIME_PARAMS[MarketRegime.SIDEWAYS])

        ctx = RegimeContext(
            regime=regime.value,
            regime_score=round(score, 3),
            trend_strength=round(trend["trend_strength"], 1),
            volatility_regime=vol["regime"],
            risk_appetite=params["risk_appetite"],
            sector_rankings=sector_rs,
            sector_momentum=sector_mom,
            recommended_position_size_pct=params["position_size_pct"],
            recommended_max_positions=params["max_positions"],
            recommended_stop_loss_pct=params["stop_loss_pct"],
            recommended_take_profit_pct=params["take_profit_pct"],
            recommended_cash_reserve_pct=params["cash_reserve_pct"],
            recommended_confidence_threshold=params["confidence_threshold"],
            spy_trend=trend["direction"],
            spy_momentum_1m=round(trend["mom_1m"], 2),
            spy_momentum_3m=round(trend["mom_3m"], 2),
            spy_above_sma50=trend["above_sma50"],
            spy_above_sma200=trend["above_sma200"],
            golden_cross=trend["golden_cross"],
        )

        self._print_summary(ctx)
        return ctx

    # ─── DATA FETCHING ──────────────────────────────────────────────────────

    def _fetch(self, symbol: str, days: int = 250) -> Optional[pd.DataFrame]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(),
            )
            bars = self.data_client.get_stock_bars(request)
            if hasattr(bars, 'df') and not bars.df.empty:
                df = bars.df.reset_index()
                if 'symbol' in df.columns:
                    df = df[df['symbol'] == symbol]
                return df
        except Exception as e:
            print(f"  ⚠️  Fetch {symbol}: {e}")
        return None

    # ─── ANALYSIS ───────────────────────────────────────────────────────────

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        close = df['close']
        price = float(close.iloc[-1])

        sma50 = close.rolling(min(50, len(close) - 1)).mean()
        sma200 = close.rolling(min(200, len(close) - 1)).mean()

        above_sma50 = price > float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else True
        above_sma200 = price > float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else True
        golden = (float(sma50.iloc[-1]) > float(sma200.iloc[-1])
                  if not pd.isna(sma50.iloc[-1]) and not pd.isna(sma200.iloc[-1]) else True)

        mom_1m = ((price / float(close.iloc[-21])) - 1) * 100 if len(close) >= 21 else 0
        mom_3m = ((price / float(close.iloc[-63])) - 1) * 100 if len(close) >= 63 else 0

        # Trend strength (directional movement ratio)
        if len(close) >= 20:
            rets = close.pct_change().dropna()
            up = rets[rets > 0].sum()
            down = abs(rets[rets < 0].sum())
            total = up + down
            strength = abs(up - down) / total * 100 if total > 0 else 0
        else:
            strength = 50

        # EMA slope
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema_slope = ((float(ema21.iloc[-1]) - float(ema21.iloc[-5])) /
                     float(ema21.iloc[-5]) * 100) if len(ema21) >= 5 else 0

        # Direction
        if above_sma50 and above_sma200 and mom_1m > 2:
            direction = "up"
        elif not above_sma50 and not above_sma200 and mom_1m < -2:
            direction = "down"
        else:
            direction = "flat"

        return {
            "price": price, "above_sma50": above_sma50, "above_sma200": above_sma200,
            "golden_cross": golden, "mom_1m": mom_1m, "mom_3m": mom_3m,
            "trend_strength": strength, "ema_slope": ema_slope, "direction": direction,
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        close, high, low = df['close'], df['high'], df['low']

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr_pct = (float(tr.rolling(14).mean().iloc[-1]) / float(close.iloc[-1])) * 100
        hist_vol = float(close.pct_change().dropna().tail(20).std() * np.sqrt(252) * 100) if len(close) >= 22 else 15

        if atr_pct < 0.8 and hist_vol < 12:
            regime = "low"
        elif atr_pct < 1.5 and hist_vol < 20:
            regime = "normal"
        elif atr_pct < 2.5 and hist_vol < 30:
            regime = "high"
        else:
            regime = "extreme"

        return {"atr_pct": atr_pct, "hist_vol": hist_vol, "regime": regime}

    def _regime_score(self, trend: Dict, vol: Dict) -> float:
        """Composite score from -1 (strong bear) to +1 (strong bull)."""
        s = 0.0

        # Trend (60%)
        if trend["above_sma50"]: s += 0.15
        if trend["above_sma200"]: s += 0.15
        if trend["golden_cross"]: s += 0.10

        # Momentum (30%)
        m1 = trend["mom_1m"]
        if m1 > 5: s += 0.15
        elif m1 > 2: s += 0.08
        elif m1 < -5: s -= 0.15
        elif m1 < -2: s -= 0.08

        m3 = trend["mom_3m"]
        if m3 > 10: s += 0.10
        elif m3 > 5: s += 0.05
        elif m3 < -10: s -= 0.10
        elif m3 < -5: s -= 0.05

        # EMA slope (5%)
        if trend["ema_slope"] > 1: s += 0.05
        elif trend["ema_slope"] < -1: s -= 0.05

        # Volatility (10%)
        if vol["regime"] == "extreme": s -= 0.10
        elif vol["regime"] == "high": s -= 0.05
        elif vol["regime"] == "low": s += 0.05

        return max(-1.0, min(1.0, s))

    def _score_to_regime(self, score: float) -> MarketRegime:
        if score >= 0.5: return MarketRegime.BULL_STRONG
        if score >= 0.2: return MarketRegime.BULL
        if score >= -0.2: return MarketRegime.SIDEWAYS
        if score >= -0.5: return MarketRegime.BEAR
        return MarketRegime.BEAR_STRONG

    # ─── SECTOR ANALYSIS ────────────────────────────────────────────────────

    def _sector_relative_strength(self) -> Dict[str, float]:
        spy = self._fetch("SPY", 60)
        if spy is None or len(spy) < 21:
            return {s: 50.0 for s in SECTOR_ETFS}

        spy_ret = ((float(spy['close'].iloc[-1]) / float(spy['close'].iloc[-21])) - 1) * 100
        rankings = {}
        for sector, etf in SECTOR_ETFS.items():
            try:
                data = self._fetch(etf, 60)
                if data is not None and len(data) >= 21:
                    ret = ((float(data['close'].iloc[-1]) / float(data['close'].iloc[-21])) - 1) * 100
                    rankings[sector] = round(max(0, min(100, 50 + (ret - spy_ret) * 5)), 1)
                else:
                    rankings[sector] = 50.0
            except Exception:
                rankings[sector] = 50.0
        return rankings

    def _sector_momentum(self) -> Dict[str, float]:
        momentum = {}
        for sector, etf in SECTOR_ETFS.items():
            try:
                data = self._fetch(etf, 60)
                if data is not None and len(data) >= 21:
                    momentum[sector] = round(
                        ((float(data['close'].iloc[-1]) / float(data['close'].iloc[-21])) - 1) * 100, 2)
                else:
                    momentum[sector] = 0.0
            except Exception:
                momentum[sector] = 0.0
        return momentum

    # ─── DISPLAY ────────────────────────────────────────────────────────────

    def _print_summary(self, ctx: RegimeContext):
        icons = {"bull_strong": "🟢🟢", "bull": "🟢", "sideways": "🟡",
                 "bear": "🔴", "bear_strong": "🔴🔴"}
        icon = icons.get(ctx.regime, "⚪")

        print(f"\n  {'=' * 60}")
        print(f"  {icon} MARKET REGIME: {ctx.regime.upper().replace('_', ' ')}")
        print(f"  {'=' * 60}")
        print(f"  Score: {ctx.regime_score:+.3f} | Trend: {ctx.trend_strength:.0f}% | Vol: {ctx.volatility_regime}")
        print(f"  SPY: {'▲' if ctx.spy_above_sma50 else '▼'} SMA50 | "
              f"{'▲' if ctx.spy_above_sma200 else '▼'} SMA200 | "
              f"{'Golden ✓' if ctx.golden_cross else 'Death ✗'}")
        print(f"  SPY Momentum: 1M {ctx.spy_momentum_1m:+.1f}% | 3M {ctx.spy_momentum_3m:+.1f}%")
        print(f"\n  📊 Adaptive Parameters:")
        print(f"     Position: {ctx.recommended_position_size_pct}% | Max: {ctx.recommended_max_positions}")
        print(f"     SL: {ctx.recommended_stop_loss_pct}% | TP: {ctx.recommended_take_profit_pct}%")
        print(f"     Cash: {ctx.recommended_cash_reserve_pct}% | Min Conf: {ctx.recommended_confidence_threshold:.0%}")

        sorted_sectors = sorted(ctx.sector_rankings.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  🏆 Top Sectors:")
        for i, (sector, rs) in enumerate(sorted_sectors[:3]):
            mom = ctx.sector_momentum.get(sector, 0)
            print(f"     {i + 1}. {sector}: RS {rs:.0f} | Mom {mom:+.1f}%")
        print(f"  {'=' * 60}")


def main():
    print_header("MARKET REGIME DETECTION")
    detector = RegimeDetector()
    ctx = detector.detect()
    save_regime_context(ctx)
    print(f"\n  💾 Regime context saved to {DATA_DIR}/regime_context.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())