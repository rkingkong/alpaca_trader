#!/usr/bin/env python3
"""
Module 1: AI-Driven Market Analysis
====================================
This module uses AI (Claude + ChatGPT) to:
1. Analyze current market conditions via web search
2. Discover the best trading opportunities across diverse sectors
3. Ensure diversification (tech, healthcare, finance, energy, consumer, crypto, etc.)
4. Perform technical analysis on AI-recommended symbols
5. Generate consensus recommendations with confidence scores

NO HARDCODED WATCHLIST - AI discovers opportunities dynamically.

Run this BEFORE market open to generate trading signals.
"""

import os
import sys
import json
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# AI API imports
import anthropic
from openai import OpenAI

# Yahoo Finance (free market data)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class AIMarketAnalyzer:
    """
    AI-driven market analyzer that discovers trading opportunities
    across diverse market segments without hardcoded watchlists.
    """
    
    # Sectors to ensure diversification
    MARKET_SECTORS = [
        "Technology",
        "Healthcare/Biotech",
        "Financial Services",
        "Energy/Oil",
        "Consumer Goods/Retail",
        "Industrial/Manufacturing",
        "Real Estate",
        "Cryptocurrency",
        "Communications",
        "Materials/Mining"
    ]
    
    def __init__(self, config_path: str = None):
        """Initialize with API credentials."""
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If no config path provided, look in script directory
        if config_path is None:
            config_path = os.path.join(self.script_dir, "config.json")
        
        self.load_config(config_path)
        self.setup_clients()
        self.output_dir = os.path.join(self.script_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load regime context if available (from Module 0)
        self.regime_context = self._load_regime_context()
        
        # Load current portfolio if available (from Module 2)
        self.portfolio_state = self._load_portfolio_state()
        
        # Load AI feedback context from Module 09 (Signal Intelligence)
        self.feedback_context = self._load_feedback_context()
    
    def _load_regime_context(self) -> Dict:
        """Load regime context from regime detector."""
        regime_path = os.path.join(self.script_dir, "data", "regime_context.json")
        if os.path.exists(regime_path):
            try:
                with open(regime_path, 'r') as f:
                    ctx = json.load(f)
                # Only use if less than 2 hours old
                ts = ctx.get("timestamp", "")
                if ts:
                    try:
                        # Handle both timezone-aware and naive timestamps
                        parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        now = datetime.now(parsed_ts.tzinfo) if parsed_ts.tzinfo else datetime.now()
                        age = (now - parsed_ts).total_seconds()
                    except Exception:
                        age = 0  # If parsing fails, use it anyway
                    if age < 7200:
                        return ctx
            except Exception:
                pass
        return {}
    
    def _load_portfolio_state(self) -> Dict:
        """Load current portfolio from Module 2 output."""
        portfolio_path = os.path.join(self.script_dir, "data", "portfolio_status.json")
        if os.path.exists(portfolio_path):
            try:
                with open(portfolio_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    

    def _load_feedback_context(self) -> Dict:
        """Load AI feedback context from Module 09 (Signal Intelligence)."""
        feedback_path = os.path.join(self.script_dir, "data", "ai_feedback_context.json")
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, 'r') as f:
                    ctx = json.load(f)
                # Check freshness (use if less than 48 hours old)
                ts = ctx.get("generated", "")
                if ts:
                    try:
                        parsed = datetime.fromisoformat(ts)
                        age_hours = (datetime.now() - parsed).total_seconds() / 3600
                        if age_hours < 48:
                            return ctx
                    except Exception:
                        return ctx
            except Exception:
                pass
        return {}

    def _get_feedback_prompt_section(self) -> str:
        """Build prompt section from Signal Intelligence feedback (Module 09)."""
        if not hasattr(self, 'feedback_context') or not self.feedback_context:
            return ""
        
        sections = []
        
        # Performance report card
        perf = self.feedback_context.get("performance_report", "")
        if perf and "Insufficient" not in perf:
            sections.append(perf)
        
        # Regime-specific constraints
        regime_prompts = self.feedback_context.get("regime_prompt_injection", {})
        if regime_prompts and self.regime_context:
            current_regime = self.regime_context.get("regime", "sideways").lower()
            matched = False
            for key in regime_prompts:
                if key in current_regime or current_regime in key:
                    sections.append(regime_prompts[key])
                    matched = True
                    break
            if not matched and "sideways" in regime_prompts:
                sections.append(regime_prompts["sideways"])
        
        # Pattern library
        pattern_text = self.feedback_context.get("pattern_prompt_injection", "")
        if pattern_text and "insufficient" not in pattern_text.lower():
            sections.append(pattern_text)
        
        # Post-mortem insights
        pm_insights = self.feedback_context.get("postmortem_insights", "")
        if pm_insights:
            sections.append(pm_insights)
        
        if not sections:
            return ""
        
        return "\n\n".join(sections) + "\n"

    def _get_regime_prompt_section(self) -> str:
        """Build a prompt section describing the current market regime."""
        if not self.regime_context:
            return ""
        
        regime = self.regime_context.get("regime", "UNKNOWN")
        score = self.regime_context.get("regime_score", 0)
        params = self.regime_context.get("adaptive_parameters", {})
        sectors = self.regime_context.get("sector_rankings", [])
        spy_data = self.regime_context.get("spy_analysis", {})
        
        # Handle sectors being either a list or dict
        if isinstance(sectors, dict):
            sectors = list(sectors.values()) if all(isinstance(v, dict) for v in sectors.values()) else [{"sector": k, **v} if isinstance(v, dict) else {"sector": k, "rs_score": v} for k, v in sectors.items()]
        
        try:
            score_str = f"{float(score):+.2f}"
        except (ValueError, TypeError):
            score_str = str(score)
        
        section = f"""
CURRENT MARKET REGIME (from quantitative regime detector):
  Regime: {regime} (score: {score_str}, range -1.0 bearish to +1.0 bullish)
  SPY above SMA50: {spy_data.get('above_sma50', 'N/A')}
  SPY above SMA200: {spy_data.get('above_sma200', 'N/A')}
  Golden Cross: {spy_data.get('golden_cross', 'N/A')}
  Volatility: {spy_data.get('volatility_regime', 'N/A')}
  Recommended position size: {params.get('position_size_pct', 'N/A')}%
  Recommended stop loss: {params.get('stop_loss_pct', 'N/A')}%
  Recommended take profit: {params.get('take_profit_pct', 'N/A')}%
"""
        if sectors:
            section += "\n  TOP SECTORS BY RELATIVE STRENGTH:\n"
            sector_list = list(sectors)[:5] if not isinstance(sectors, list) else sectors[:5]
            for i, s in enumerate(sector_list):
                if isinstance(s, dict):
                    name = s.get('sector', s.get('name', 'N/A'))
                    rs = s.get('rs_score', s.get('rs', 0))
                    mom = s.get('momentum_1m', s.get('momentum', 0))
                    try:
                        section += f"    {i+1}. {name}: RS={rs}, Momentum={float(mom):+.1f}%\n"
                    except (ValueError, TypeError):
                        section += f"    {i+1}. {name}: RS={rs}, Momentum={mom}\n"
                else:
                    section += f"    {i+1}. {s}\n"
        
        section += f"""
  REGIME STRATEGY GUIDANCE:
"""
        if "BULL" in regime:
            section += """    - Be AGGRESSIVE: ride momentum, use wider stops, let winners run
    - Overweight top RS sectors (Energy, Staples currently leading)
    - BUY breakouts and pullbacks to support in strong names
    - Minimum 2:1 reward-to-risk, aim for 3:1 on high conviction"""
        elif "BEAR" in regime:
            section += """    - Be DEFENSIVE: smaller positions, tighter stops, more cash
    - Focus on short opportunities and defensive sectors (Utilities, Staples, Healthcare)
    - Only BUY highest-conviction oversold bounces with clear catalysts
    - Consider inverse ETFs and protective puts"""
        else:
            section += """    - Be SELECTIVE: focus on sector leaders, avoid laggards
    - Tighter position sizing, demand strong catalysts for new entries
    - Range-trade: buy support, sell resistance"""
        
        return section
    
    def _get_portfolio_prompt_section(self) -> str:
        """Build a prompt section describing current portfolio holdings."""
        if not self.portfolio_state:
            return ""
        
        positions = self.portfolio_state.get("positions", [])
        if not positions:
            return "\nCURRENT PORTFOLIO: No positions held (100% cash available)\n"
        
        # Handle positions being a dict (keyed by symbol) or a list
        if isinstance(positions, dict):
            pos_list = list(positions.values())
        else:
            pos_list = list(positions)
        
        section = "\nCURRENT PORTFOLIO HOLDINGS (consider for SELL recommendations):\n"
        for pos in pos_list[:15]:  # Limit to avoid token bloat
            if not isinstance(pos, dict):
                continue
            symbol = pos.get("symbol", "?")
            pnl_pct = pos.get("unrealized_plpc", 0)
            value = pos.get("market_value", 0)
            try:
                pnl_pct = float(pnl_pct)
                value = float(value)
                pnl_icon = "+" if pnl_pct >= 0 else ""
                section += f"  {symbol}: ${value:,.0f} ({pnl_icon}{pnl_pct:.1f}%)\n"
            except (ValueError, TypeError):
                section += f"  {symbol}: value={value}, pnl={pnl_pct}\n"
        
        section += """
  → Recommend SELL for positions that are: underperforming, technically broken, 
    or in weak sectors. Don't just recommend buying more of what we already own.
  → Focus new BUY picks on symbols NOT already in the portfolio.
"""
        return section
        
    def load_config(self, config_path: str):
        """Load API credentials from config file or environment."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.alpaca_key = config.get('ALPACA_API_KEY', os.getenv('ALPACA_API_KEY'))
            self.alpaca_secret = config.get('ALPACA_SECRET_KEY', os.getenv('ALPACA_SECRET_KEY'))
            self.claude_key = config.get('ANTHROPIC_API_KEY', os.getenv('ANTHROPIC_API_KEY'))
            self.openai_key = config.get('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
            self.newsapi_key = config.get('NEWSAPI_KEY', os.getenv('NEWSAPI_KEY'))
            self.finnhub_key = config.get('FINNHUB_API_KEY', os.getenv('FINNHUB_API_KEY'))
            self.fred_key = config.get('FRED_API_KEY', os.getenv('FRED_API_KEY'))
            self.alpha_vantage_key = config.get('ALPHA_VANTAGE_KEY', os.getenv('ALPHA_VANTAGE_KEY'))
            self.paper_trading = config.get('PAPER_TRADING', True)
        else:
            self.alpaca_key = os.getenv('ALPACA_API_KEY')
            self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
            self.claude_key = os.getenv('ANTHROPIC_API_KEY')
            self.openai_key = os.getenv('OPENAI_API_KEY')
            self.newsapi_key = os.getenv('NEWSAPI_KEY')
            self.finnhub_key = os.getenv('FINNHUB_API_KEY')
            self.fred_key = os.getenv('FRED_API_KEY')
            self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
            self.paper_trading = True
            
    def setup_clients(self):
        """Initialize API clients."""
        self.trading_client = TradingClient(
            self.alpaca_key, 
            self.alpaca_secret, 
            paper=self.paper_trading
        )
        self.stock_data_client = StockHistoricalDataClient(
            self.alpaca_key, 
            self.alpaca_secret
        )
        self.crypto_data_client = CryptoHistoricalDataClient(
            self.alpaca_key, 
            self.alpaca_secret
        )
        
        if self.claude_key:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_key)
        else:
            self.claude_client = None
            print("Ã¢Å¡Â Ã¯Â¸Â  Claude API key not configured")
            
        if self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
        else:
            self.openai_client = None
            print("Ã¢Å¡Â Ã¯Â¸Â  OpenAI API key not configured")

    # =========================================================================
    # MARKET DATA GATHERING
    # =========================================================================
    
    def fetch_market_news(self) -> List[Dict]:
        """Fetch general market news and trends."""
        articles = []
        
        if self.newsapi_key:
            try:
                queries = [
                    "stock market today",
                    "best performing stocks",
                    "market movers today",
                    "cryptocurrency market",
                    "sector rotation"
                ]
                
                for query in queries[:3]:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": query,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "pageSize": 5,
                        "apiKey": self.newsapi_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for article in data.get('articles', [])[:3]:
                            articles.append({
                                "title": article.get('title', ''),
                                "description": article.get('description', ''),
                                "source": article.get('source', {}).get('name', ''),
                                "query": query
                            })
            except Exception as e:
                print(f"  News fetch error: {e}")
        
        return articles
    
    # =========================================================================
    # FINNHUB MARKET INTELLIGENCE
    # =========================================================================
    
    def fetch_finnhub_market_news(self) -> List[Dict]:
        """Fetch general market news from Finnhub (higher quality financial news)."""
        if not self.finnhub_key:
            return []
        
        articles = []
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                news = response.json()
                for article in news[:12]:  # Top 12 most recent
                    articles.append({
                        "title": article.get("headline", ""),
                        "description": article.get("summary", "")[:200],
                        "source": article.get("source", ""),
                        "query": "finnhub_general"
                    })
        except Exception as e:
            print(f"  Finnhub news error: {e}")
        
        return articles
    
    def fetch_finnhub_earnings_calendar(self) -> List[Dict]:
        """Fetch upcoming earnings for the next 7 days."""
        if not self.finnhub_key:
            return []
        
        earnings = []
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {"from": today, "to": next_week, "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for entry in data.get("earningsCalendar", [])[:30]:
                    earnings.append({
                        "symbol": entry.get("symbol", ""),
                        "date": entry.get("date", ""),
                        "hour": entry.get("hour", ""),  # bmo=before market, amc=after market
                        "eps_estimate": entry.get("epsEstimate"),
                        "revenue_estimate": entry.get("revenueEstimate"),
                    })
        except Exception as e:
            print(f"  Finnhub earnings error: {e}")
        
        return earnings
    
    def fetch_finnhub_analyst_ratings(self, symbol: str) -> Dict:
        """Fetch analyst recommendation trends for a symbol."""
        if not self.finnhub_key:
            return {}
        
        try:
            url = "https://finnhub.io/api/v1/stock/recommendation"
            params = {"symbol": symbol, "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                recs = response.json()
                if recs and len(recs) > 0:
                    latest = recs[0]  # Most recent month
                    total = (latest.get("buy", 0) + latest.get("hold", 0) + 
                            latest.get("sell", 0) + latest.get("strongBuy", 0) + 
                            latest.get("strongSell", 0))
                    if total > 0:
                        bullish = latest.get("strongBuy", 0) + latest.get("buy", 0)
                        bearish = latest.get("strongSell", 0) + latest.get("sell", 0)
                        return {
                            "period": latest.get("period", ""),
                            "strong_buy": latest.get("strongBuy", 0),
                            "buy": latest.get("buy", 0),
                            "hold": latest.get("hold", 0),
                            "sell": latest.get("sell", 0),
                            "strong_sell": latest.get("strongSell", 0),
                            "bullish_pct": round(bullish / total * 100, 1),
                            "consensus": "bullish" if bullish > bearish + latest.get("hold", 0) 
                                        else "bearish" if bearish > bullish 
                                        else "mixed"
                        }
        except Exception:
            pass
        return {}
    
    def fetch_finnhub_insider_sentiment(self, symbol: str) -> Dict:
        """Fetch insider sentiment (buying vs selling) for a symbol."""
        if not self.finnhub_key:
            return {}
        
        try:
            url = "https://finnhub.io/api/v1/stock/insider-sentiment"
            params = {
                "symbol": symbol, 
                "from": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                "to": datetime.now().strftime('%Y-%m-%d'),
                "token": self.finnhub_key
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                sentiments = data.get("data", [])
                if sentiments:
                    # Aggregate last 3 months
                    total_mspr = sum(s.get("mspr", 0) for s in sentiments[-3:])
                    return {
                        "mspr_3m": round(total_mspr, 2),  # Monthly Share Purchase Ratio
                        "signal": "buying" if total_mspr > 10 else "selling" if total_mspr < -10 else "neutral"
                    }
        except Exception:
            pass
        return {}
    
    def _build_finnhub_context(self, finnhub_news: List, earnings: List) -> str:
        """Build a text section with Finnhub intelligence for AI prompts."""
        sections = []
        
        if finnhub_news:
            news_text = "\n".join(f"  - {n['title']}" for n in finnhub_news[:8])
            sections.append(f"FINANCIAL NEWS (Finnhub - high quality):\n{news_text}")
        
        if earnings:
            # Filter to well-known symbols
            notable = [e for e in earnings if len(e.get("symbol", "")) <= 5][:10]
            if notable:
                earn_text = "\n".join(
                    f"  - {e['symbol']}: {e['date']} ({e.get('hour', '?')}) | "
                    f"EPS est: {e.get('eps_estimate', 'N/A')}"
                    for e in notable
                )
                sections.append(f"UPCOMING EARNINGS (next 7 days):\n{earn_text}\n"
                              f"  → Consider pre-earnings positioning on high-conviction names\n"
                              f"  → Avoid new entries right before earnings unless thesis is catalyst-dependent")
        
        return "\n\n".join(sections)
    
    # =========================================================================
    # FRED - MACROECONOMIC DATA
    # =========================================================================
    
    def fetch_fred_macro_data(self) -> Dict:
        """Fetch key macroeconomic indicators from FRED (Federal Reserve)."""
        if not self.fred_key:
            return {}
        
        macro = {}
        # Key series: Fed Funds, 10Y yield, 2Y yield, VIX, Unemployment, CPI
        series_map = {
            "DFF": "fed_funds_rate",
            "DGS10": "treasury_10y",
            "DGS2": "treasury_2y",
            "VIXCLS": "vix",
            "UNRATE": "unemployment_rate",
            "ICSA": "initial_claims",
            "T10Y2Y": "yield_curve_spread",
        }
        
        for series_id, label in series_map.items():
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                    "observation_start": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                }
                response = requests.get(url, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get("observations", [])
                    # Get most recent non-"." value
                    for obs in observations:
                        val = obs.get("value", ".")
                        if val != ".":
                            macro[label] = {
                                "value": float(val),
                                "date": obs.get("date", "")
                            }
                            break
            except Exception as e:
                pass  # Don't let one failed series kill the whole fetch
        
        # Calculate yield curve inversion status
        if "treasury_10y" in macro and "treasury_2y" in macro:
            spread = macro["treasury_10y"]["value"] - macro["treasury_2y"]["value"]
            macro["yield_curve_calc"] = {
                "spread": round(spread, 3),
                "inverted": spread < 0,
                "signal": "recession_warning" if spread < -0.2 else "caution" if spread < 0.1 else "normal"
            }
        
        return macro
    
    def _build_fred_context(self, macro: Dict) -> str:
        """Build prompt section from FRED macro data."""
        if not macro:
            return ""
        
        lines = ["MACROECONOMIC CONDITIONS (FRED - Federal Reserve data):"]
        
        if "fed_funds_rate" in macro:
            lines.append(f"  Fed Funds Rate: {macro['fed_funds_rate']['value']:.2f}%")
        if "treasury_10y" in macro:
            lines.append(f"  10Y Treasury Yield: {macro['treasury_10y']['value']:.2f}%")
        if "treasury_2y" in macro:
            lines.append(f"  2Y Treasury Yield: {macro['treasury_2y']['value']:.2f}%")
        if "yield_curve_calc" in macro:
            yc = macro["yield_curve_calc"]
            status = "INVERTED ⚠️" if yc["inverted"] else "Normal"
            lines.append(f"  Yield Curve Spread (10Y-2Y): {yc['spread']:+.3f}% ({status})")
        elif "yield_curve_spread" in macro:
            val = macro["yield_curve_spread"]["value"]
            status = "INVERTED" if val < 0 else "Normal"
            lines.append(f"  Yield Curve Spread: {val:+.3f}% ({status})")
        if "vix" in macro:
            vix = macro["vix"]["value"]
            vix_label = "EXTREME FEAR" if vix > 30 else "HIGH" if vix > 20 else "MODERATE" if vix > 15 else "LOW/COMPLACENT"
            lines.append(f"  VIX (Fear Index): {vix:.1f} ({vix_label})")
        if "unemployment_rate" in macro:
            lines.append(f"  Unemployment Rate: {macro['unemployment_rate']['value']:.1f}%")
        if "initial_claims" in macro:
            claims = macro["initial_claims"]["value"]
            lines.append(f"  Initial Jobless Claims: {claims:,.0f}")
        
        # Add interpretation guidance
        lines.append("")
        lines.append("  MACRO INTERPRETATION:")
        
        vix_val = macro.get("vix", {}).get("value", 15)
        ff_val = macro.get("fed_funds_rate", {}).get("value", 5)
        yc_inverted = macro.get("yield_curve_calc", {}).get("inverted", False)
        
        if vix_val > 25:
            lines.append("    - High VIX = fear/uncertainty → wider stops needed, consider protective puts")
        elif vix_val < 14:
            lines.append("    - Low VIX = complacency → market may be stretched, be selective")
        
        if yc_inverted:
            lines.append("    - Inverted yield curve = classic recession signal → favor defensives, shorter time horizons")
        
        if ff_val > 5:
            lines.append("    - High interest rates = headwind for growth stocks → favor value, cash-flow-positive names")
        elif ff_val < 2:
            lines.append("    - Low rates = tailwind for growth/tech → be aggressive on momentum names")
        
        return "\n".join(lines)
    
    # =========================================================================
    # ALPHA VANTAGE - SECTOR PERFORMANCE & FUNDAMENTALS
    # =========================================================================
    
    def fetch_alpha_vantage_sector_performance(self) -> Dict:
        """Fetch real-time sector performance from Alpha Vantage."""
        if not self.alpha_vantage_key:
            return {}
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "SECTOR",
                "apikey": self.alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = {}
                
                # Get different timeframes
                for period_key, period_label in [
                    ("Rank A: Real-Time Performance", "realtime"),
                    ("Rank B: 1 Day Performance", "1day"),
                    ("Rank C: 5 Day Performance", "5day"),
                    ("Rank D: 1 Month Performance", "1month"),
                    ("Rank E: 3 Month Performance", "3month"),
                ]:
                    period_data = data.get(period_key, {})
                    if period_data:
                        result[period_label] = {
                            sector: float(pct.replace('%', ''))
                            for sector, pct in period_data.items()
                            if pct and '%' in str(pct)
                        }
                
                return result
        except Exception as e:
            print(f"  Alpha Vantage sector error: {e}")
        return {}
    
    def _build_alpha_vantage_context(self, sector_data: Dict) -> str:
        """Build prompt section from Alpha Vantage sector performance."""
        if not sector_data:
            return ""
        
        lines = ["SECTOR PERFORMANCE (Alpha Vantage - real-time):"]
        
        # Show 1-month performance sorted by strength
        monthly = sector_data.get("1month", {})
        if monthly:
            sorted_sectors = sorted(monthly.items(), key=lambda x: x[1], reverse=True)
            lines.append("  1-Month Sector Performance (strongest to weakest):")
            for sector, pct in sorted_sectors:
                icon = "▲" if pct > 0 else "▼"
                lines.append(f"    {icon} {sector}: {pct:+.2f}%")
            
            # Identify rotation
            top_3 = [s[0] for s in sorted_sectors[:3]]
            bottom_3 = [s[0] for s in sorted_sectors[-3:]]
            lines.append(f"\n  Money flowing INTO: {', '.join(top_3)}")
            lines.append(f"  Money flowing OUT OF: {', '.join(bottom_3)}")
        
        # Show 5-day for short-term momentum
        weekly = sector_data.get("5day", {})
        if weekly:
            sorted_weekly = sorted(weekly.items(), key=lambda x: x[1], reverse=True)
            lines.append("\n  5-Day Sector Momentum (short-term):")
            for sector, pct in sorted_weekly[:5]:
                lines.append(f"    {sector}: {pct:+.2f}%")
        
        return "\n".join(lines)
    
    # =========================================================================
    # YAHOO FINANCE - FREE FUNDAMENTALS & ANALYTICS
    # =========================================================================
    
    def fetch_yahoo_market_movers(self) -> Dict:
        """Fetch market movers, fear/greed indicators from Yahoo Finance."""
        if not HAS_YFINANCE:
            return {}
        
        movers = {}
        try:
            # Get major index performance with more detail
            tickers = yf.Tickers("SPY QQQ IWM DIA XLF XLE XLK XLV XLP XLI XLB XLRE XLC XLU XLY")
            
            for symbol in ["SPY", "QQQ", "IWM", "DIA"]:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        chg = ((current - prev) / prev) * 100
                        movers[symbol] = {
                            "price": round(float(current), 2),
                            "daily_change_pct": round(chg, 2),
                            "volume": int(hist['Volume'].iloc[-1]),
                            "avg_volume": int(hist['Volume'].mean())
                        }
                except Exception:
                    pass
            
            # Get sector ETF performance for rotation analysis
            sector_etfs = {
                "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
                "XLV": "Healthcare", "XLP": "Consumer Staples", "XLI": "Industrials",
                "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communications",
                "XLU": "Utilities", "XLY": "Consumer Discretionary"
            }
            
            sector_perf = {}
            for etf, name in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1mo")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        month_ago = hist['Close'].iloc[0]
                        chg_1m = ((current - month_ago) / month_ago) * 100
                        daily_chg = ((current - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                        sector_perf[name] = {
                            "etf": etf,
                            "price": round(float(current), 2),
                            "change_1m": round(chg_1m, 2),
                            "change_1d": round(daily_chg, 2),
                        }
                except Exception:
                    pass
            
            if sector_perf:
                movers["sector_performance"] = sector_perf
                
        except Exception as e:
            print(f"  Yahoo Finance movers error: {e}")
        
        return movers
    
    def fetch_yahoo_stock_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamentals for a specific stock via Yahoo Finance."""
        if not HAS_YFINANCE:
            return {}
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                return {}
            
            fundamentals = {}
            
            # Key valuation metrics
            for key, label in [
                ("trailingPE", "pe_ratio"),
                ("forwardPE", "forward_pe"),
                ("priceToBook", "price_to_book"),
                ("marketCap", "market_cap"),
                ("enterpriseValue", "enterprise_value"),
                ("trailingEps", "eps"),
                ("forwardEps", "forward_eps"),
                ("dividendYield", "dividend_yield"),
                ("beta", "beta"),
                ("shortPercentOfFloat", "short_interest"),
                ("targetMeanPrice", "analyst_target"),
                ("currentPrice", "current_price"),
                ("recommendationKey", "recommendation"),
                ("numberOfAnalystOpinions", "analyst_count"),
                ("revenueGrowth", "revenue_growth"),
                ("earningsGrowth", "earnings_growth"),
                ("profitMargins", "profit_margin"),
                ("returnOnEquity", "roe"),
                ("debtToEquity", "debt_to_equity"),
                ("freeCashflow", "free_cash_flow"),
            ]:
                val = info.get(key)
                if val is not None:
                    fundamentals[label] = val
            
            # Calculate upside to analyst target
            if "analyst_target" in fundamentals and "current_price" in fundamentals:
                target = fundamentals["analyst_target"]
                current = fundamentals["current_price"]
                if current > 0:
                    fundamentals["upside_pct"] = round(((target - current) / current) * 100, 1)
            
            return fundamentals
            
        except Exception:
            return {}
    
    def _build_yahoo_market_context(self, movers: Dict) -> str:
        """Build prompt section from Yahoo Finance market data."""
        if not movers:
            return ""
        
        lines = ["MARKET OVERVIEW (Yahoo Finance - real-time):"]
        
        # Index summary
        for symbol in ["SPY", "QQQ", "IWM", "DIA"]:
            if symbol in movers:
                d = movers[symbol]
                vol_ratio = d.get("volume", 0) / max(d.get("avg_volume", 1), 1)
                vol_tag = " HIGH VOL" if vol_ratio > 1.5 else ""
                lines.append(f"  {symbol}: ${d['price']:.2f} ({d['daily_change_pct']:+.2f}%){vol_tag}")
        
        # Sector rotation analysis
        sector_perf = movers.get("sector_performance", {})
        if sector_perf:
            sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1].get("change_1m", 0), reverse=True)
            
            lines.append("\n  SECTOR ROTATION (1-Month, Yahoo Finance):")
            for name, data in sorted_sectors:
                icon = "▲" if data["change_1m"] > 0 else "▼"
                lines.append(f"    {icon} {name:22} ({data['etf']}): 1M {data['change_1m']:+.1f}% | Today {data['change_1d']:+.1f}%")
            
            # Identify leaders and laggards
            leaders = [s[0] for s in sorted_sectors[:3]]
            laggards = [s[0] for s in sorted_sectors[-3:]]
            lines.append(f"\n  → SECTOR LEADERS: {', '.join(leaders)}")
            lines.append(f"  → SECTOR LAGGARDS: {', '.join(laggards)}")
            lines.append(f"  → Overweight leaders, underweight laggards for sector rotation alpha")
        
        return "\n".join(lines)
    
    def _build_macro_intelligence(self, fred_data: Dict, av_sector: Dict, yahoo_movers: Dict) -> str:
        """Combine all macro/market intelligence into a single context block for AI prompts."""
        sections = []
        
        fred_ctx = self._build_fred_context(fred_data)
        if fred_ctx:
            sections.append(fred_ctx)
        
        # Prefer Yahoo sector data (free, more reliable), fall back to Alpha Vantage
        yahoo_ctx = self._build_yahoo_market_context(yahoo_movers)
        if yahoo_ctx:
            sections.append(yahoo_ctx)
        elif av_sector:
            av_ctx = self._build_alpha_vantage_context(av_sector)
            if av_ctx:
                sections.append(av_ctx)
        
        if not sections:
            return ""
        
        return "\n\n".join(sections)
    
    def fetch_market_data_summary(self) -> Dict:
        """Fetch market indices and crypto data for context."""
        summary = {
            "indices": {},
            "crypto": {},
            "timestamp": datetime.now().isoformat()
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Get major indices via Alpaca
        index_symbols = ["SPY", "QQQ", "IWM", "DIA"]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=index_symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = self.stock_data_client.get_stock_bars(request)
            
            if hasattr(bars, 'df') and not bars.df.empty:
                df = bars.df.reset_index()
                for symbol in index_symbols:
                    symbol_data = df[df['symbol'] == symbol]
                    if not symbol_data.empty:
                        latest = symbol_data.iloc[-1]
                        prev = symbol_data.iloc[-2] if len(symbol_data) > 1 else latest
                        change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100
                        summary["indices"][symbol] = {
                            "price": float(latest['close']),
                            "change_pct": round(change_pct, 2)
                        }
        except Exception as e:
            print(f"  Index data error: {e}")
        
        # Get major crypto
        crypto_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=crypto_symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = self.crypto_data_client.get_crypto_bars(request)
            
            if hasattr(bars, 'df') and not bars.df.empty:
                df = bars.df.reset_index()
                for symbol in crypto_symbols:
                    symbol_data = df[df['symbol'] == symbol]
                    if not symbol_data.empty:
                        latest = symbol_data.iloc[-1]
                        prev = symbol_data.iloc[-2] if len(symbol_data) > 1 else latest
                        change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100
                        summary["crypto"][symbol] = {
                            "price": float(latest['close']),
                            "change_pct": round(change_pct, 2)
                        }
        except Exception as e:
            print(f"  Crypto data error: {e}")
        
        return summary

    # =========================================================================
    # AI STOCK DISCOVERY
    # =========================================================================
    
    def discover_opportunities_claude(self, market_context: Dict, news: List[Dict], finnhub_context: str = "", macro_context: str = "") -> List[Dict]:
        """Use Claude to discover trading opportunities across diverse sectors."""
        if not self.claude_client:
            return []
        
        news_text = "\n".join([f"- {n['title']}" for n in news[:10]]) if news else "No news available"
        
        indices_text = ""
        for symbol, data in market_context.get("indices", {}).items():
            indices_text += f"  {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
        
        crypto_text = ""
        for symbol, data in market_context.get("crypto", {}).items():
            crypto_text += f"  {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
        
        system_prompt = """You are an elite quantitative portfolio manager running an aggressive growth fund targeting 15-20% annualized returns.
Your approach combines:
1. MOMENTUM INVESTING (like TQQQ/TECL leveraged strategies): Buy strength, ride trends
2. MACRO REGIME AWARENESS (like Bridgewater Pure Alpha): Adapt to market conditions
3. SECTOR ROTATION (like Discovery Capital +52% in 2024): Overweight leading sectors
4. SYSTEMATIC SIGNALS (like Quantedge ~20% CAGR): RSI power zone, volume confirmation, trend alignment

Your goal is to MAXIMIZE returns while managing risk through diversification and position sizing, not by avoiding opportunity.

KEY PRINCIPLES:
- Momentum is your friend: stocks that are going up tend to keep going up
- Volatility is opportunity, not risk Ã¢â‚¬â€ wider moves mean bigger profits with proper stops
- Catalysts drive outsized returns: earnings, FDA approvals, product launches, macro shifts
- Asymmetric risk/reward is the goal: risk $1 to make $3+
- Crypto is a legitimate asset class with massive return potential
- Be bold on conviction but diversified across sectors
- A 5% stop loss with a 15% take profit is better than a 2% stop with 5% target"""

        # Build regime-aware and portfolio-aware context
        regime_section = self._get_regime_prompt_section()
        portfolio_section = self._get_portfolio_prompt_section()
        feedback_section = self._get_feedback_prompt_section()

        prompt = f"""CURRENT MARKET CONDITIONS:
Date: {datetime.now().strftime('%Y-%m-%d')}

Major Indices:
{indices_text if indices_text else "  Data unavailable"}

Cryptocurrency:
{crypto_text if crypto_text else "  Data unavailable"}

Recent Market Headlines:
{news_text}
{regime_section}
{portfolio_section}
{feedback_section}
{finnhub_context}
{macro_context}
YOUR TASK:
Identify 15-20 of the HIGHEST ALPHA trading opportunities right now. Think like a hedge fund manager who needs to beat the market by a wide margin. Focus on:

1. MOMENTUM PLAYS: What's moving WITH strong volume? Breakouts above key resistance?
2. CATALYST-DRIVEN: Earnings beats, FDA decisions, M&A rumors, product launches, macro policy shifts
3. SECTOR ROTATION: Where is institutional money flowing? Which sectors are early in their cycle?
4. MEAN REVERSION: Quality names that are deeply oversold with clear recovery catalysts
5. CRYPTO OPPORTUNITIES: BTC, ETH, SOL, and others with strong technical or fundamental setups
6. SHORT OPPORTUNITIES: What's overextended, losing momentum, or facing headwinds?

DIVERSIFICATION Ã¢â‚¬â€ include opportunities from ALL of these sectors:
1. Technology (AAPL, MSFT, NVDA, AMD, GOOGL, META, SMCI, PLTR, etc.)
2. Healthcare/Biotech (JNJ, PFE, UNH, MRNA, ABBV, LLY, ISRG, etc.)
3. Financial Services (JPM, BAC, GS, V, MA, COIN, SOFI, etc.)
4. Energy (XOM, CVX, OXY, FSLR, ENPH, CEG, VST, etc.)
5. Consumer/Retail (AMZN, WMT, COST, NKE, SBUX, LULU, etc.)
6. Industrial/Manufacturing (CAT, DE, BA, HON, GE, RTX, etc.)
7. Cryptocurrency (BTC/USD, ETH/USD, SOL/USD, AVAX/USD, DOGE/USD, LINK/USD)
8. Communications/Media (DIS, NFLX, T, VZ, SPOT, RBLX, etc.)
9. Real Estate (O, AMT, PLD, SPG, EQIX, DLR, etc.)
10. Materials/Mining (FCX, NEM, LIN, APD, NUE, GOLD, etc.)

CONVICTION LEVELS Ã¢â‚¬â€ be honest and aggressive:
- 0.85-1.0: "I would put my own money on this TODAY" Ã¢â‚¬â€ multiple strong catalysts aligning
- 0.70-0.84: Strong setup, high probability trade with clear thesis
- 0.55-0.69: Good opportunity but some uncertainty remains
- Below 0.55: WATCH only Ã¢â‚¬â€ interesting but not actionable yet

STOP LOSS / TAKE PROFIT GUIDANCE:
- For momentum stocks: 4-6% stop, 10-20% take profit (ride the trend)
- For value/oversold bounces: 5-8% stop, 8-15% take profit
- For crypto: 7-12% stop, 15-30% take profit (wider for higher vol)
- ALWAYS aim for minimum 2:1 reward-to-risk ratio

Respond ONLY with a JSON array:
[
    {{
        "symbol": "TICKER",
        "asset_type": "stock" or "crypto",
        "sector": "sector name from list above",
        "action": "BUY" or "SELL" or "WATCH",
        "confidence": 0.0 to 1.0,
        "conviction_tier": "high" or "medium" or "base",
        "reasoning": "2-3 sentence explanation with specific data points",
        "catalyst": "specific near-term catalyst with expected timing",
        "time_horizon": "days" or "weeks" or "months",
        "risk_level": "low" or "medium" or "high",
        "suggested_stop_loss_pct": number,
        "suggested_take_profit_pct": number
    }}
]

IMPORTANT:
- For crypto, use format: BTC/USD, ETH/USD, SOL/USD, etc.
- Only recommend liquid, easily tradeable symbols on major exchanges
- Be SPECIFIC: cite price levels, dates, earnings estimates, technical patterns
- Don't shy away from high-conviction aggressive calls Ã¢â‚¬â€ that's what we want
- Include at least 3-4 picks with confidence above 0.80"""

        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                opportunities = json.loads(json_match.group())
                return opportunities
            return []
            
        except Exception as e:
            print(f"  Claude discovery error: {e}")
            return []
    
    def discover_opportunities_chatgpt(self, market_context: Dict, news: List[Dict], finnhub_context: str = "", macro_context: str = "") -> List[Dict]:
        """Use ChatGPT to discover trading opportunities across diverse sectors."""
        if not self.openai_client:
            return []
        
        news_text = "\n".join([f"- {n['title']}" for n in news[:10]]) if news else "No news available"
        
        indices_text = ""
        for symbol, data in market_context.get("indices", {}).items():
            indices_text += f"  {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
        
        crypto_text = ""
        for symbol, data in market_context.get("crypto", {}).items():
            crypto_text += f"  {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)\n"
        
        gpt_system = """You are a quantitative trading strategist who specializes in technical analysis, price action, and institutional money flow. Your analytical edge comes from reading the tape - understanding WHERE SMART MONEY IS MOVING before the crowd catches on.

YOUR APPROACH:
- Price action and volume tell the truth - news and narratives follow price
- Institutional accumulation/distribution patterns reveal the next big moves
- Relative strength analysis: find what outperforms its sector and the market
- Mean reversion works for quality - oversold names with strong fundamentals bounce hard
- Crypto follows cyclical patterns and correlates with risk appetite
- Short setups are as valuable as longs - profit from both directions
- Asymmetric risk/reward: minimum 2:1 ratio on every trade"""

        # Build regime-aware and portfolio-aware context for GPT too
        regime_section = self._get_regime_prompt_section()
        portfolio_section = self._get_portfolio_prompt_section()
        feedback_section = self._get_feedback_prompt_section()

        prompt = f"""MARKET SNAPSHOT:
Date: {datetime.now().strftime('%Y-%m-%d')}

Index Performance:
{indices_text if indices_text else "  Data unavailable"}

Crypto Markets:
{crypto_text if crypto_text else "  Data unavailable"}

Headlines:
{news_text}
{regime_section}
{portfolio_section}
{feedback_section}
{finnhub_context}
{macro_context}
ANALYSIS REQUEST:
As a quant strategist, identify 15-20 HIGH-CONVICTION trades. Use your technical and flow analysis lens:

1. BREAKOUT SETUPS: Stocks consolidating near resistance with rising volume
2. MOMENTUM CONTINUATION: Stocks with strong relative strength vs SPY/QQQ
3. EARNINGS PLAYS: Pre-earnings positioning on high-probability names
4. REVERSAL TRADES: RSI divergence, hammer candles, or capitulation volume
5. CRYPTO TECHNICAL: Chart patterns, on-chain flows, ETF inflow trends
6. SHORT SETUPS: Bearish divergence, head-and-shoulders, declining momentum

SECTOR COVERAGE (at least one each):
1. Technology (AAPL, MSFT, NVDA, AMD, GOOGL, META, SMCI, PLTR, etc.)
2. Healthcare/Biotech (JNJ, PFE, UNH, MRNA, LLY, ISRG, etc.)
3. Financial Services (JPM, GS, V, MA, COIN, SOFI, etc.)
4. Energy (XOM, CVX, OXY, FSLR, CEG, VST, etc.)
5. Consumer/Retail (AMZN, WMT, COST, SBUX, LULU, etc.)
6. Industrial/Manufacturing (CAT, DE, BA, HON, GE, RTX, etc.)
7. Cryptocurrency (BTC/USD, ETH/USD, SOL/USD, AVAX/USD, DOGE/USD, LINK/USD)
8. Communications/Media (DIS, NFLX, SPOT, RBLX, etc.)
9. Real Estate (O, AMT, PLD, SPG, EQIX, DLR, etc.)
10. Materials/Mining (FCX, NEM, LIN, APD, GOLD, etc.)

CONVICTION SCALE:
- 0.85-1.0: Multiple technicals + catalyst = trade of the week
- 0.70-0.84: Strong setup with clear risk/reward
- 0.55-0.69: Decent setup, needs confirmation
- Below 0.55: Watchlist only

RISK:
- Stocks: 4-7% stop, 10-20% target
- Crypto: 7-12% stop, 15-30% target
- Minimum 2:1 reward-to-risk on every pick

Respond ONLY with JSON array:
[
    {{
        "symbol": "TICKER",
        "asset_type": "stock" or "crypto",
        "sector": "sector name",
        "action": "BUY" or "SELL" or "WATCH",
        "confidence": 0.0 to 1.0,
        "conviction_tier": "high" or "medium" or "base",
        "reasoning": "cite specific technical levels, patterns, or data",
        "catalyst": "catalyst with timing",
        "time_horizon": "days" or "weeks" or "months",
        "risk_level": "low" or "medium" or "high",
        "suggested_stop_loss_pct": number,
        "suggested_take_profit_pct": number
    }}
]

CRITICAL:
- Crypto: BTC/USD, ETH/USD, SOL/USD format
- Cite specific support/resistance levels
- Include contrarian picks where risk/reward is exceptional
- At least 3-4 picks above 0.80 confidence
- Be differentiated - don't just pick obvious blue chips"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": gpt_system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=6000
            )
            
            response_text = response.choices[0].message.content
            
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                opportunities = json.loads(json_match.group())
                return opportunities
            return []
            
        except Exception as e:
            print(f"  ChatGPT discovery error: {e}")
            return []
    
    def merge_ai_discoveries(self, claude_picks: List[Dict], chatgpt_picks: List[Dict]) -> List[Dict]:
        """Merge and score discoveries from both AI models."""
        merged = {}
        
        # Process Claude picks
        for pick in claude_picks:
            symbol = pick.get("symbol", "").upper()
            if not symbol:
                continue
            merged[symbol] = {
                **pick,
                "symbol": symbol,
                "claude_confidence": pick.get("confidence", 0.5),
                "claude_action": pick.get("action", "HOLD"),
                "claude_reasoning": pick.get("reasoning", ""),
                "chatgpt_confidence": 0,
                "chatgpt_action": None,
                "chatgpt_reasoning": "",
                "sources": ["claude"]
            }
        
        # Process ChatGPT picks
        for pick in chatgpt_picks:
            symbol = pick.get("symbol", "").upper()
            if not symbol:
                continue
            
            if symbol in merged:
                # Both AIs picked this symbol!
                merged[symbol]["chatgpt_confidence"] = pick.get("confidence", 0.5)
                merged[symbol]["chatgpt_action"] = pick.get("action", "HOLD")
                merged[symbol]["chatgpt_reasoning"] = pick.get("reasoning", "")
                merged[symbol]["sources"].append("chatgpt")
                
                # Use ChatGPT's stop/take profit if Claude didn't provide
                if not merged[symbol].get("suggested_stop_loss_pct"):
                    merged[symbol]["suggested_stop_loss_pct"] = pick.get("suggested_stop_loss_pct", 2.0)
                if not merged[symbol].get("suggested_take_profit_pct"):
                    merged[symbol]["suggested_take_profit_pct"] = pick.get("suggested_take_profit_pct", 4.0)
            else:
                merged[symbol] = {
                    **pick,
                    "symbol": symbol,
                    "claude_confidence": 0,
                    "claude_action": None,
                    "claude_reasoning": "",
                    "chatgpt_confidence": pick.get("confidence", 0.5),
                    "chatgpt_action": pick.get("action", "HOLD"),
                    "chatgpt_reasoning": pick.get("reasoning", ""),
                    "sources": ["chatgpt"]
                }
        
        # Calculate consensus
        for symbol, data in merged.items():
            claude_conf = data.get("claude_confidence", 0)
            chatgpt_conf = data.get("chatgpt_confidence", 0)
            claude_action = data.get("claude_action")
            chatgpt_action = data.get("chatgpt_action")
            
            if len(data["sources"]) == 2:
                # Both AIs agree on symbol
                # Weighted consensus using Signal Intelligence feedback
                _weights = self.feedback_context.get("consensus_weights", {}) if hasattr(self, 'feedback_context') else {}
                _w_claude = _weights.get("claude_weight", 0.5)
                _w_gpt = _weights.get("gpt_weight", 0.5)
                _bonus = _weights.get("consensus_bonus_pct", 10) / 100
                consensus_confidence = min(1.0, (claude_conf * _w_claude + gpt_conf * _w_gpt) + _bonus)
                
                # Check if they agree on action too
                if claude_action == chatgpt_action:
                    data["consensus_confidence"] = min(avg_conf * 1.4, 1.0)  # 40% bonus for full agreement
                    data["action"] = claude_action
                    data["ai_agreement"] = "full"
                else:
                    # Disagree on action - use higher confidence one but penalize
                    data["consensus_confidence"] = avg_conf * 0.8  # Both AIs noticed it - that's still signal
                    data["action"] = claude_action if claude_conf > chatgpt_conf else chatgpt_action
                    data["ai_agreement"] = "partial"
                
                # Combine reasoning
                data["reasoning"] = f"Claude: {data['claude_reasoning']} | ChatGPT: {data['chatgpt_reasoning']}"
            else:
                # Only one source
                data["consensus_confidence"] = max(claude_conf, chatgpt_conf) * 0.88  # 12% penalty for single source (one strong opinion is still valuable)
                data["action"] = data.get("action") or claude_action or chatgpt_action or "WATCH"
                data["ai_agreement"] = "single"
        
        # Convert to list and sort
        result = list(merged.values())
        result.sort(key=lambda x: x.get("consensus_confidence", 0), reverse=True)
        
        return result

    # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range for volatility."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
    
    def get_technical_analysis(self, symbol: str, asset_type: str = "stock") -> Dict:
        """Get technical analysis for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=250)  # 250 days for proper SMA200
            
            if asset_type == "crypto":
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                bars = self.crypto_data_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                bars = self.stock_data_client.get_stock_bars(request)
            
            df = bars.df.reset_index() if hasattr(bars, 'df') else pd.DataFrame()
            
            if df.empty or len(df) < 26:
                return {"error": f"Insufficient data for {symbol}", "symbol": symbol}
            
            close_prices = df['close']
            current_price = close_prices.iloc[-1]
            
            rsi = self.calculate_rsi(close_prices)
            macd, macd_signal, macd_histogram = self.calculate_macd(close_prices)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
            atr = self.calculate_atr(df['high'], df['low'], close_prices)
            
            atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
            
            # EMA crossover (9/21)
            ema_9 = close_prices.ewm(span=9, adjust=False).mean()
            ema_21 = close_prices.ewm(span=21, adjust=False).mean()
            ema_cross_bullish = ema_9.iloc[-1] > ema_21.iloc[-1]
            
            # SMA 50/200 trend alignment
            sma_50 = close_prices.rolling(window=min(50, len(close_prices)-1)).mean()
            sma_200 = close_prices.rolling(window=min(200, len(close_prices)-1)).mean()
            above_sma50 = current_price > sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else False
            above_sma200 = current_price > sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else False
            golden_cross = sma_50.iloc[-1] > sma_200.iloc[-1] if (not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1])) else False
            
            # Price momentum (1-month return)
            if len(close_prices) >= 21:
                momentum_1m = ((current_price - close_prices.iloc[-21]) / close_prices.iloc[-21]) * 100
            else:
                momentum_1m = 0.0
            
            # Relative Strength vs SPY (critical momentum indicator)
            rs_vs_spy = 0.0
            try:
                if asset_type != "crypto":
                    spy_request = StockBarsRequest(
                        symbol_or_symbols="SPY",
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    spy_bars = self.stock_data_client.get_stock_bars(spy_request)
                    spy_df = spy_bars.df.reset_index() if hasattr(spy_bars, 'df') else pd.DataFrame()
                    if not spy_df.empty and len(spy_df) >= 21:
                        spy_close = spy_df['close']
                        spy_mom = ((spy_close.iloc[-1] - spy_close.iloc[-21]) / spy_close.iloc[-21]) * 100
                        rs_vs_spy = momentum_1m - spy_mom  # Positive = outperforming SPY
            except Exception:
                rs_vs_spy = 0.0
            
            # Volume analysis
            if 'volume' in df.columns:
                vol_sma = df['volume'].rolling(window=20).mean()
                current_vol = df['volume'].iloc[-1]
                avg_vol = vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else current_vol
                volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                high_volume = volume_ratio > 1.5  # 50% above average
            else:
                volume_ratio = 1.0
                high_volume = False
            
            # Technical signals (MOMENTUM-FOCUSED)
            # RSI 50-70 = power zone for momentum stocks
            # Below 30 = falling knife, NOT a buy signal
            signals = {
                "rsi_signal": "weak" if rsi < 35 else "recovering" if rsi < 50 else "momentum" if rsi < 70 else "extended" if rsi < 80 else "overextended",
                "macd_signal": "bullish" if macd_histogram > 0 else "bearish",
                "bb_signal": "breakdown" if current_price < bb_lower else "breakout" if current_price > bb_upper else "neutral",
                "ema_signal": "bullish" if ema_cross_bullish else "bearish",
                "volume_signal": "confirming" if high_volume else "normal"
            }
            
            # Count bullish/bearish signals (MOMENTUM APPROACH)
            bullish_count = sum([
                signals["rsi_signal"] in ["momentum", "extended"],  # RSI power zone
                signals["macd_signal"] == "bullish",
                signals["bb_signal"] == "breakout",                  # Upper BB = strength
                signals["ema_signal"] == "bullish",
                signals["volume_signal"] == "confirming" and signals["macd_signal"] == "bullish"
            ])
            
            bearish_count = sum([
                signals["rsi_signal"] == "weak",                     # Low RSI = falling knife
                signals["macd_signal"] == "bearish",
                signals["bb_signal"] == "breakdown",                 # Lower BB = weakness
                signals["ema_signal"] == "bearish",
                signals["volume_signal"] == "confirming" and signals["macd_signal"] == "bearish"
            ])
            
            technical_bias = "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "neutral"
            
            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "indicators": {
                    "rsi": round(rsi, 2),
                    "macd": round(macd, 4),
                    "macd_signal": round(macd_signal, 4),
                    "macd_histogram": round(macd_histogram, 4),
                    "bb_upper": round(bb_upper, 2),
                    "bb_middle": round(bb_middle, 2),
                    "bb_lower": round(bb_lower, 2),
                    "atr": round(atr, 4),
                    "atr_percent": round(atr_percent, 2),
                    "ema_9": round(float(ema_9.iloc[-1]), 2),
                    "ema_21": round(float(ema_21.iloc[-1]), 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "above_sma50": above_sma50,
                    "above_sma200": above_sma200,
                    "golden_cross": golden_cross,
                    "momentum_1m_pct": round(momentum_1m, 2),
                    "rs_vs_spy": round(rs_vs_spy, 2)
                },
                "signals": signals,
                "technical_bias": technical_bias,
                "trend_aligned": above_sma50 and above_sma200,
                "volatility": "high" if atr_percent > 3 else "medium" if atr_percent > 1.5 else "low"
            }
            
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    # =========================================================================
    # FINAL RECOMMENDATION SCORING
    # =========================================================================
    
    def score_opportunity(self, discovery: Dict, technical: Dict) -> Dict:
        """Combine AI discovery with technical analysis for final scoring."""
        
        ai_confidence = discovery.get("consensus_confidence", 0.5)
        ai_action = discovery.get("action", "HOLD")
        ai_agreement = discovery.get("ai_agreement", "single")
        
        tech_bias = technical.get("technical_bias", "neutral")
        volatility = technical.get("volatility", "medium")
        trend_aligned = technical.get("trend_aligned", False)
        indicators = technical.get("indicators", {})
        
        # Alignment bonus/penalty
        if ai_action == "BUY" and tech_bias == "bullish":
            alignment_factor = 1.2
        elif ai_action == "SELL" and tech_bias == "bearish":
            alignment_factor = 1.2
        elif ai_action == "BUY" and tech_bias == "bearish":
            alignment_factor = 0.80
        elif ai_action == "SELL" and tech_bias == "bullish":
            alignment_factor = 0.80
        else:
            alignment_factor = 1.0
        
        # TREND ALIGNMENT BONUS
        if trend_aligned and ai_action == "BUY":
            trend_factor = 1.15  # 15% boost for confirmed uptrend
        elif not indicators.get("above_sma200", True) and ai_action == "BUY":
            trend_factor = 0.75  # 25% penalty for below 200 SMA
        else:
            trend_factor = 1.0
        
        # MOMENTUM BONUS
        momentum_1m = indicators.get("momentum_1m_pct", 0)
        if momentum_1m > 5 and ai_action == "BUY":
            momentum_factor = 1.10  # Strong recent momentum
        elif momentum_1m < -5 and ai_action == "BUY":
            momentum_factor = 0.85  # Negative momentum = caution
        else:
            momentum_factor = 1.0
        
        # RELATIVE STRENGTH VS SPY BONUS (NEW)
        rs_vs_spy = indicators.get("rs_vs_spy", 0)
        if rs_vs_spy > 3 and ai_action == "BUY":
            rs_factor = 1.08  # Outperforming SPY by 3%+ in last month
        elif rs_vs_spy < -5 and ai_action == "BUY":
            rs_factor = 0.90  # Significantly underperforming SPY
        else:
            rs_factor = 1.0
        
        # REGIME ALIGNMENT (NEW)
        regime_factor = 1.0
        if self.regime_context:
            regime = self.regime_context.get("regime", "")
            # Check if pick aligns with sector rankings
            raw_sectors = self.regime_context.get("sector_rankings", [])
            if isinstance(raw_sectors, dict):
                raw_sectors = [{"sector": k, **v} if isinstance(v, dict) else {"sector": k} for k, v in raw_sectors.items()]
            top_sectors = [s.get("sector", s.get("name", "")).lower() for s in raw_sectors[:3] if isinstance(s, dict)]
            pick_sector = discovery.get("sector", "").lower()
            
            sector_is_leader = any(ts in pick_sector or pick_sector in ts for ts in top_sectors)
            
            if "BULL" in regime and ai_action == "BUY":
                regime_factor = 1.08  # Bull regime favors buys
                if sector_is_leader:
                    regime_factor = 1.15  # Extra bonus for top-sector picks
            elif "BEAR" in regime and ai_action == "BUY":
                regime_factor = 0.85  # Bear regime penalizes buys
            elif "BEAR" in regime and ai_action == "SELL":
                regime_factor = 1.10  # Bear regime favors sells
        
        # Volatility adjustment
        volatility_factor = 0.95 if volatility == "high" else 1.0
        
        # Final score
        final_confidence = min(
            ai_confidence * alignment_factor * volatility_factor * trend_factor 
            * momentum_factor * rs_factor * regime_factor, 
            1.0
        )
        
        # Determine final action
        if final_confidence >= 0.45:
            final_action = ai_action
        else:
            final_action = "WATCH"
        
        # Determine if regime-aligned and sector-aligned
        regime_aligned = regime_factor > 1.0
        sector_rs_aligned = rs_factor > 1.0
        
        return {
            "symbol": discovery.get("symbol"),
            "asset_type": discovery.get("asset_type", "stock"),
            "sector": discovery.get("sector", "Unknown"),
            "action": final_action,
            "confidence": round(final_confidence, 2),
            "ai_agreement": ai_agreement,
            "technical_bias": tech_bias,
            "volatility": volatility,
            "reasoning": discovery.get("reasoning", ""),
            "catalyst": discovery.get("catalyst", ""),
            "risk_level": discovery.get("risk_level", "medium"),
            "current_price": technical.get("current_price"),
            "indicators": technical.get("indicators", {}),
            "stop_loss_pct": discovery.get("suggested_stop_loss_pct", 2.0),
            "take_profit_pct": discovery.get("suggested_take_profit_pct", 4.0),
            "sources": discovery.get("sources", []),
            "regime_aligned": regime_aligned,
            "sector_rs_aligned": sector_rs_aligned,
            "rs_vs_spy": rs_vs_spy,
        }

    # =========================================================================
    # MAIN ANALYSIS FLOW
    # =========================================================================
    
    def run_full_analysis(self) -> Dict:
        """Run complete AI-driven market analysis."""
        print("=" * 70)
        print("AI-DRIVEN MARKET ANALYSIS - Module 1")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\nÃ°Å¸Â¤â€“ AI will discover the best opportunities across all sectors...")
        print("   No hardcoded watchlist - fully dynamic discovery\n")
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "market_context": {},
            "discoveries": [],
            "recommendations": [],
            "buy_signals": [],
            "sell_signals": [],
            "watch_signals": [],
            "sector_coverage": {}
        }
        
        # Step 1: Gather market context
        print("Ã°Å¸â€œÅ  Step 1: Gathering market data...")
        market_context = self.fetch_market_data_summary()
        results["market_context"] = market_context
        
        print("Ã°Å¸â€œÂ° Step 2: Fetching market news...")
        news = self.fetch_market_news()
        print(f"   Found {len(news)} news articles")

        # Step 2b: Finnhub market intelligence
        print("   Fetching Finnhub financial news...")
        finnhub_news = self.fetch_finnhub_market_news()
        print(f"   Found {len(finnhub_news)} Finnhub articles")
        
        print("   Fetching upcoming earnings calendar...")
        earnings_calendar = self.fetch_finnhub_earnings_calendar()
        print(f"   Found {len(earnings_calendar)} upcoming earnings")
        
        # Merge all news sources (Finnhub first = higher quality)
        all_news = finnhub_news + news
        finnhub_context = self._build_finnhub_context(finnhub_news, earnings_calendar)
        results["finnhub_context"] = {
            "news_count": len(finnhub_news),
            "earnings_count": len(earnings_calendar),
            "earnings": earnings_calendar[:15]
        }
        
        # Step 2c: Macro & Sector Intelligence
        print("\n   Fetching macro data (FRED)...")
        fred_data = self.fetch_fred_macro_data()
        fred_items = len(fred_data)
        print(f"   FRED indicators loaded: {fred_items}")
        
        print("   Fetching sector performance (Alpha Vantage)...")
        av_sector = self.fetch_alpha_vantage_sector_performance()
        av_items = len(av_sector)
        print(f"   Alpha Vantage sector periods: {av_items}")
        
        print("   Fetching market data (Yahoo Finance)...")
        yahoo_movers = self.fetch_yahoo_market_movers()
        yahoo_sectors = len(yahoo_movers.get("sector_performance", {}))
        print(f"   Yahoo Finance: {len(yahoo_movers)} indices, {yahoo_sectors} sector ETFs")
        
        # Build combined macro intelligence for AI prompts
        macro_context = self._build_macro_intelligence(fred_data, av_sector, yahoo_movers)
        results["macro_data"] = {
            "fred_indicators": fred_items,
            "av_sectors": av_items,
            "yahoo_sectors": yahoo_sectors,
            "fred_data": fred_data,
        }
        
        # Step 2: AI Discovery
        print("\nÃ°Å¸Â¤â€“ Step 3: AI Discovery Phase...")
        
        print("   Ã°Å¸â€Âµ Querying Claude for opportunities...")
        claude_picks = self.discover_opportunities_claude(market_context, all_news, finnhub_context, macro_context)
        print(f"      Claude discovered {len(claude_picks)} opportunities")
        
        print("   Ã°Å¸Å¸Â¢ Querying ChatGPT for opportunities...")
        chatgpt_picks = self.discover_opportunities_chatgpt(market_context, all_news, finnhub_context, macro_context)
        print(f"      ChatGPT discovered {len(chatgpt_picks)} opportunities")
        
        # Step 3: Merge discoveries
        print("\nÃ°Å¸â€â€ž Step 4: Merging AI recommendations...")
        merged_discoveries = self.merge_ai_discoveries(claude_picks, chatgpt_picks)
        results["discoveries"] = merged_discoveries
        print(f"   Total unique opportunities: {len(merged_discoveries)}")
        
        # Count agreements
        full_agreements = sum(1 for d in merged_discoveries if d.get("ai_agreement") == "full")
        partial_agreements = sum(1 for d in merged_discoveries if d.get("ai_agreement") == "partial")
        print(f"   Full AI agreement: {full_agreements} symbols")
        print(f"   Partial agreement: {partial_agreements} symbols")
        
        # Step 4: Technical analysis
        print("\nÃ°Å¸â€œË† Step 5: Technical analysis on discovered symbols...")
        
        # -- Filter out OCC option symbols (AI sometimes recommends them) --
        _occ_pattern = re.compile(r'^[A-Z]{1,6}\d{6}[CP]\d{8}$')
        _pre_count = len(merged_discoveries)
        merged_discoveries = [
            d for d in merged_discoveries
            if not _occ_pattern.match(d.get("symbol", ""))
        ]
        _filtered = _pre_count - len(merged_discoveries)
        if _filtered > 0:
            print(f"   Filtered {_filtered} option-contract symbol(s) "
                  f"(route to Options Engine)")

        for i, discovery in enumerate(merged_discoveries):
            symbol = discovery.get("symbol")
            asset_type = discovery.get("asset_type", "stock")

            print(f"   [{i+1}/{len(merged_discoveries)}] Analyzing {symbol}...", end=" ")
            technical = self.get_technical_analysis(symbol, asset_type)
            
            if "error" in technical:
                print(f"Ã¢Å¡Â Ã¯Â¸Â  {technical['error'][:30]}")
                scored = {
                    **discovery,
                    "current_price": None,
                    "technical_error": technical.get("error"),
                    "action": "WATCH",
                    "confidence": discovery.get("consensus_confidence", 0.5) * 0.6,
                    "indicators": {}
                }
            else:
                scored = self.score_opportunity(discovery, technical)
                
                # Enrich with Finnhub analyst ratings for stocks (not crypto)
                if asset_type == "stock" and self.finnhub_key and scored.get("confidence", 0) >= 0.45:
                    analyst = self.fetch_finnhub_analyst_ratings(symbol)
                    if analyst:
                        scored["analyst_consensus"] = analyst.get("consensus", "")
                        scored["analyst_bullish_pct"] = analyst.get("bullish_pct", 0)
                        # Analyst alignment bonus
                        if scored["action"] == "BUY" and analyst.get("consensus") == "bullish":
                            scored["confidence"] = min(round(scored["confidence"] * 1.05, 2), 1.0)
                        elif scored["action"] == "BUY" and analyst.get("consensus") == "bearish":
                            scored["confidence"] = round(scored["confidence"] * 0.95, 2)

                # Enrich with Yahoo Finance fundamentals for high-confidence stocks
                if asset_type == "stock" and HAS_YFINANCE and scored.get("confidence", 0) >= 0.55:
                    yf_fund = self.fetch_yahoo_stock_fundamentals(symbol)
                    if yf_fund:
                        scored["pe_ratio"] = yf_fund.get("pe_ratio")
                        scored["forward_pe"] = yf_fund.get("forward_pe")
                        scored["analyst_target"] = yf_fund.get("analyst_target")
                        scored["upside_pct"] = yf_fund.get("upside_pct")
                        scored["short_interest"] = yf_fund.get("short_interest")
                        scored["revenue_growth"] = yf_fund.get("revenue_growth")
                        scored["earnings_growth"] = yf_fund.get("earnings_growth")
                        scored["recommendation"] = yf_fund.get("recommendation", "")
                        
                        # Upside alignment: if analyst target confirms AI BUY
                        upside = yf_fund.get("upside_pct", 0)
                        if scored["action"] == "BUY" and upside and upside > 15:
                            scored["confidence"] = min(round(scored["confidence"] * 1.05, 2), 1.0)
                        elif scored["action"] == "BUY" and upside and upside < -5:
                            scored["confidence"] = round(scored["confidence"] * 0.92, 2)
                print(f"Ã¢Å“â€œ ${technical.get('current_price', 0):.2f}")
            
            results["recommendations"].append(scored)
            
            # Categorize
            action = scored.get("action", "WATCH")
            if action == "BUY":
                results["buy_signals"].append(scored)
            elif action == "SELL":
                results["sell_signals"].append(scored)
            else:
                results["watch_signals"].append(scored)
            
            # Track sectors
            sector = scored.get("sector", "Unknown")
            if sector not in results["sector_coverage"]:
                results["sector_coverage"][sector] = []
            results["sector_coverage"][sector].append(symbol)
        
        # Sort by confidence
        results["buy_signals"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        results["sell_signals"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        results["recommendations"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Summary
        results["summary"] = {
            "total_discovered": len(merged_discoveries),
            "full_ai_agreements": full_agreements,
            "partial_ai_agreements": partial_agreements,
            "buy_signals_count": len(results["buy_signals"]),
            "sell_signals_count": len(results["sell_signals"]),
            "watch_signals_count": len(results["watch_signals"]),
            "sectors_covered": len(results["sector_coverage"]),
            "top_buys": [s["symbol"] for s in results["buy_signals"][:3]],
            "top_sells": [s["symbol"] for s in results["sell_signals"][:3]]
        }
        
        # Save (with numpy type handling)
        output_file = os.path.join(self.output_dir, "recommendations.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_default)
        
        # Print summary
        try:
            self._print_summary(results)
        except Exception as e:
            print(f"\n  Warning: Summary display error: {e}")
            print(f"  (Results still saved to {output_file})")
        
        return results
    
    @staticmethod
    def _json_default(obj):
        """Handle numpy types and other non-serializable objects."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)
    
    def _print_summary(self, results: Dict):
        """Print detailed analysis summary with AI reasoning."""
        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)
        
        summary = results.get("summary", {})
        
        # Show regime context if available
        if self.regime_context:
            regime = self.regime_context.get("regime", "UNKNOWN")
            score = self.regime_context.get("regime_score", 0)
            try:
                print(f"\n  Regime: {regime} (score: {float(score):+.2f})")
            except (ValueError, TypeError):
                print(f"\n  Regime: {regime} (score: {score})")
            raw_sectors = self.regime_context.get("sector_rankings", [])
            if isinstance(raw_sectors, dict):
                raw_sectors = [{"sector": k, **v} if isinstance(v, dict) else {"sector": k} for k, v in raw_sectors.items()]
            sectors = raw_sectors[:3] if isinstance(raw_sectors, list) else []
            if sectors:
                sector_str = ", ".join(f"{s.get('sector', s.get('name', '?'))}" if isinstance(s, dict) else str(s) for s in sectors)
                print(f"  Leading Sectors: {sector_str}")
        
        print(f"\n  DISCOVERY SUMMARY:")
        print(f"   Total opportunities discovered: {summary.get('total_discovered', 0)}")
        print(f"   Full AI agreement (both engines): {summary.get('full_ai_agreements', 0)}")
        print(f"   Partial agreement: {summary.get('partial_ai_agreements', 0)}")
        print(f"   Sectors covered: {summary.get('sectors_covered', 0)}")
        
        # Detailed BUY signals
        buy_signals = results.get("buy_signals", [])
        print(f"\n  BUY SIGNALS ({len(buy_signals)}):")
        print(f"  {'Symbol':8} {'Conf':>5} {'Agree':>7} {'Tech':>8} {'RS/SPY':>7} {'SL%':>5} {'TP%':>5} {'Sector':20}")
        print(f"  {'-'*75}")
        
        for sig in buy_signals[:12]:
            agree_str = sig.get("ai_agreement", "?")[:6]
            tech_str = sig.get("technical_bias", "?")[:7]
            rs = sig.get("rs_vs_spy") or (sig.get("indicators") or {}).get("rs_vs_spy", 0) or 0
            sl = sig.get("stop_loss_pct") or 0
            tp = sig.get("take_profit_pct") or 0
            sector = sig.get("sector", "?")[:19]
            regime_tag = " R" if sig.get("regime_aligned") else ""
            
            print(f"  {sig['symbol']:8} {(sig.get('confidence') or 0):>5.0%} {agree_str:>7} {tech_str:>8} {rs:>+6.1f}% {sl:>4.0f}% {tp:>4.0f}% {sector}{regime_tag}")
            
            # Show catalyst and analyst info for top picks
            details = []
            if sig.get('catalyst') and sig['confidence'] >= 0.60:
                details.append(f"Catalyst: {sig['catalyst'][:55]}")
            if sig.get('analyst_consensus'):
                try:
                    details.append(f"Analysts: {sig['analyst_consensus']} ({float(sig.get('analyst_bullish_pct') or 0):.0f}% bullish)")
                except (ValueError, TypeError):
                    details.append(f"Analysts: {sig['analyst_consensus']}")
            # Yahoo Finance fundamentals
            yf_parts = []
            if sig.get('pe_ratio') is not None:
                try:
                    yf_parts.append(f"PE:{float(sig['pe_ratio']):.1f}")
                except (ValueError, TypeError):
                    pass
            if sig.get('upside_pct') is not None:
                try:
                    yf_parts.append(f"Target upside:{float(sig['upside_pct']):+.0f}%")
                except (ValueError, TypeError):
                    pass
            if sig.get('short_interest'):
                si = sig['short_interest']
                if si > 0.1:  # >10% short interest is notable
                    yf_parts.append(f"Short:{si:.0%}")
            if yf_parts:
                details.append(" | ".join(yf_parts))
            if details:
                print(f"           -> {' | '.join(details)}")
        
        if len(buy_signals) > 12:
            print(f"  ... and {len(buy_signals) - 12} more")
        
        # SELL signals
        sell_signals = results.get("sell_signals", [])
        if sell_signals:
            print(f"\n  SELL SIGNALS ({len(sell_signals)}):")
            for sig in sell_signals[:5]:
                print(f"  {sig['symbol']:8} {(sig.get('confidence') or 0):>5.0%} | {sig.get('reasoning', '')[:60]}")
        
        # WATCH list (brief)
        watch_signals = results.get("watch_signals", [])
        if watch_signals:
            watch_str = ", ".join(s['symbol'] for s in watch_signals[:8])
            print(f"\n  WATCH ({len(watch_signals)}): {watch_str}")
        
        # Macro data summary
        macro_data = results.get("macro_data", {})
        if macro_data:
            fred_data = macro_data.get("fred_data") or {}
            print(f"\n  MACRO ENVIRONMENT:")
            if "fed_funds_rate" in fred_data:
                print(f"   Fed Funds: {fred_data['fed_funds_rate']['value']:.2f}%", end="")
            if "vix" in fred_data:
                vix = fred_data['vix']['value']
                vix_tag = "FEAR" if vix > 25 else "HIGH" if vix > 20 else "NORMAL" if vix > 15 else "LOW"
                print(f" | VIX: {vix:.1f} ({vix_tag})", end="")
            if fred_data.get("yield_curve_calc"):
                yc = fred_data["yield_curve_calc"]
                print(f" | Yield Curve: {yc['spread']:+.2f}% {'(INVERTED)' if yc['inverted'] else ''}", end="")
            print()  # newline
        
        # Data sources summary
        print(f"\n  DATA SOURCES USED:")
        sources = []
        if (results.get("finnhub_context") or {}).get("news_count", 0) > 0:
            sources.append(f"Finnhub ({(results.get('finnhub_context') or {}).get('news_count', 0)} news, {(results.get('finnhub_context') or {}).get('earnings_count', 0)} earnings)")
        if macro_data.get("fred_indicators", 0) > 0:
            sources.append(f"FRED ({macro_data['fred_indicators']} macro indicators)")
        if macro_data.get("av_sectors", 0) > 0:
            sources.append(f"Alpha Vantage ({macro_data['av_sectors']} sector periods)")
        if macro_data.get("yahoo_sectors", 0) > 0:
            sources.append(f"Yahoo Finance ({macro_data['yahoo_sectors']} sectors)")
        sources.extend(["Alpaca (price data)", "Claude AI", "GPT-4o"])
        for s in sources:
            print(f"   - {s}")
        
        # Sector coverage
        print(f"\n  SECTOR COVERAGE:")
        for sector, symbols in results.get("sector_coverage", {}).items():
            print(f"   {sector[:25]:25} -> {', '.join(symbols[:4])}")
        
        # Scoring legend
        print(f"\n  LEGEND: Conf=Final Confidence | Agree=AI Agreement | Tech=Technical Bias")
        print(f"          RS/SPY=Relative Strength vs SPY | R=Regime Aligned")
        print(f"\n  Results saved to: data/recommendations.json")

def main():
    """Main entry point."""
    try:
        analyzer = AIMarketAnalyzer()
        results = analyzer.run_full_analysis()
        
        if not results:
            print("\n⚠️  Analysis completed but returned no results")
            return 1
        
        buys = len(results.get("buy_signals", []))
        sells = len(results.get("sell_signals", []))
        
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"Ã°Å¸Å½Â¯ READY FOR EXECUTION")
        print(f"   {buys} BUY signals | {sells} SELL signals")
        print(f"{sep}")
        
        return 0
    except Exception as e:
        print(f"\nÃ¢ÂÅ’ Market analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())