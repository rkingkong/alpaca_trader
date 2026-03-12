#!/usr/bin/env python3
"""
Module 11: News & Market Intelligence Aggregator
==================================================
Multi-source news aggregator with RSS feeds, market sentiment, earnings
calendar, economic events, and AI-driven sentiment scoring.

Sources:
  1. RSS Feeds       — Reuters, Bloomberg (via RSS), MarketWatch, CNBC, Seeking Alpha
  2. Finnhub News    — High-quality financial news API
  3. FRED Events     — Economic indicator releases
  4. Earnings Cal    — Upcoming earnings from Finnhub
  5. SEC Filings     — Notable 13F, 8-K filings (via RSS)
  6. Crypto News     — CoinDesk, CoinTelegraph RSS
  7. Sector Movers   — Alpha Vantage sector performance

Output:
  data/market_intel.json       — Aggregated intelligence feed
  data/news_sentiment.json     — Sentiment scores & trend
  data/earnings_calendar.json  — Upcoming earnings events
  data/economic_calendar.json  — Upcoming economic releases

Usage:
  python 11_news_aggregator.py --collect       # Collect from all sources
  python 11_news_aggregator.py --sentiment     # Score sentiment on collected news
  python 11_news_aggregator.py --earnings      # Refresh earnings calendar
  python 11_news_aggregator.py --economic      # Refresh economic calendar
  python 11_news_aggregator.py --full          # All of the above
  python 11_news_aggregator.py --daemon        # Run on schedule (every 30 min)
"""

import os
import sys
import json
import time
import hashlib
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

import requests

try:
    from config import Config, DATA_DIR, save_json, print_header
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config, DATA_DIR, save_json, print_header


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RSS_FEEDS = {
    # --- General Market ---
    "reuters_markets": {
        "url": "https://feeds.reuters.com/reuters/businessNews",
        "category": "macro",
        "priority": 1,
    },
    "cnbc_top": {
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "category": "macro",
        "priority": 2,
    },
    "marketwatch_top": {
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "category": "macro",
        "priority": 2,
    },
    "wsj_markets": {
        "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "category": "macro",
        "priority": 1,
    },
    # --- Technology ---
    "techcrunch": {
        "url": "https://techcrunch.com/feed/",
        "category": "tech",
        "priority": 3,
    },
    # --- Crypto ---
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "category": "crypto",
        "priority": 2,
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "category": "crypto",
        "priority": 2,
    },
    # --- SEC / Regulatory ---
    "sec_filings": {
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&search_text=&action=getcurrent&output=atom",
        "category": "regulatory",
        "priority": 3,
    },
    # --- Seeking Alpha ---
    "seekingalpha_market": {
        "url": "https://seekingalpha.com/market_currents.xml",
        "category": "analysis",
        "priority": 2,
    },
}

# Keyword -> sentiment mapping for rule-based scoring
BULLISH_KEYWORDS = [
    "surge", "rally", "beat", "record", "upgrade", "bullish", "growth",
    "outperform", "breakout", "accelerate", "soar", "profit", "gains",
    "strong", "exceed", "optimis", "buy", "upside", "boom", "expand",
    "positive", "recovery", "momentum", "highs", "all-time",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "miss", "downgrade", "bearish", "recession",
    "decline", "sell-off", "selloff", "warning", "risk", "fear",
    "loss", "weak", "drop", "fall", "cut", "layoff", "default",
    "inflation", "overvalued", "bubble", "concern", "threat", "slump",
]

# Sectors and their associated keywords for tagging
SECTOR_KEYWORDS = {
    "Technology": ["tech", "ai", "semiconductor", "chip", "software", "cloud", "nvidia", "apple", "microsoft", "google", "meta", "amazon"],
    "Healthcare": ["pharma", "biotech", "drug", "fda", "health", "medical", "vaccine", "hospital"],
    "Financial": ["bank", "fed", "interest rate", "treasury", "yield", "loan", "mortgage", "goldman", "jpmorgan"],
    "Energy": ["oil", "gas", "energy", "opec", "crude", "solar", "wind", "renewable", "exxon"],
    "Crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi", "nft", "stablecoin"],
    "Consumer": ["retail", "consumer", "spending", "walmart", "amazon", "costco", "target"],
    "Industrial": ["manufacturing", "pmi", "industrial", "supply chain", "logistics", "caterpillar"],
    "Real Estate": ["housing", "real estate", "mortgage", "reit", "property", "home sales"],
}

# FRED economic indicator series for calendar
FRED_SERIES = {
    "UNRATE": {"name": "Unemployment Rate", "category": "labor", "freq": "monthly"},
    "CPIAUCSL": {"name": "CPI (Consumer Prices)", "category": "inflation", "freq": "monthly"},
    "GDP": {"name": "GDP Growth Rate", "category": "growth", "freq": "quarterly"},
    "FEDFUNDS": {"name": "Fed Funds Rate", "category": "monetary", "freq": "daily"},
    "T10Y2Y": {"name": "10Y-2Y Yield Spread", "category": "bonds", "freq": "daily"},
    "VIXCLS": {"name": "VIX Volatility Index", "category": "volatility", "freq": "daily"},
    "DCOILWTICO": {"name": "WTI Crude Oil Price", "category": "commodities", "freq": "daily"},
    "DGS10": {"name": "10-Year Treasury Yield", "category": "bonds", "freq": "daily"},
    "ICSA": {"name": "Initial Jobless Claims", "category": "labor", "freq": "weekly"},
    "RSXFS": {"name": "Retail Sales", "category": "consumer", "freq": "monthly"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NewsItem:
    title: str
    source: str
    category: str
    url: str = ""
    description: str = ""
    published: str = ""
    sentiment: str = "neutral"   # bullish / bearish / neutral
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sectors: list = None
    priority: int = 2
    hash_id: str = ""

    def __post_init__(self):
        if self.sectors is None:
            self.sectors = []
        if not self.hash_id:
            self.hash_id = hashlib.md5(f"{self.title}{self.source}".encode()).hexdigest()[:12]


@dataclass
class EarningsEvent:
    symbol: str
    date: str
    hour: str = ""  # bmo (before market open), amc (after market close)
    eps_estimate: float = 0.0
    eps_actual: float = 0.0
    revenue_estimate: float = 0.0
    surprise_pct: float = 0.0


@dataclass
class EconomicEvent:
    name: str
    series_id: str
    category: str
    latest_value: float = 0.0
    previous_value: float = 0.0
    last_updated: str = ""
    change_pct: float = 0.0
    trend: str = "stable"  # rising / falling / stable


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class NewsAggregator:
    """Multi-source news collection, deduplication, and sentiment scoring."""

    def __init__(self):
        self.cfg = Config()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.items: List[NewsItem] = []
        self._seen_hashes = set()

    # ─── RSS Feed Collection ────────────────────────────────────────────────

    def collect_rss_feeds(self) -> int:
        """Collect news from all configured RSS feeds."""
        print("\n  Collecting RSS feeds...")
        collected = 0

        for feed_name, feed_cfg in RSS_FEEDS.items():
            try:
                resp = requests.get(feed_cfg["url"], timeout=10, headers={
                    "User-Agent": "LuxverumCapital/2.0 NewsAggregator"
                })
                if resp.status_code != 200:
                    print(f"    [{feed_name}] HTTP {resp.status_code} — skipped")
                    continue

                items = self._parse_rss(resp.text, feed_name, feed_cfg)
                for item in items:
                    if item.hash_id not in self._seen_hashes:
                        self._seen_hashes.add(item.hash_id)
                        self.items.append(item)
                        collected += 1

                print(f"    [{feed_name}] {len(items)} items")
            except Exception as e:
                print(f"    [{feed_name}] Error: {str(e)[:60]}")

        return collected

    def _parse_rss(self, xml_text: str, feed_name: str, feed_cfg: Dict) -> List[NewsItem]:
        """Parse RSS/Atom XML into NewsItem objects."""
        items = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return items

        # Handle both RSS 2.0 and Atom formats
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "dc": "http://purl.org/dc/elements/1.1/",
        }

        # Try RSS 2.0 first
        rss_items = root.findall(".//item")
        if not rss_items:
            # Try Atom
            rss_items = root.findall(".//atom:entry", namespaces)

        for entry in rss_items[:15]:  # Max 15 per feed
            title = self._get_text(entry, ["title", "atom:title"], namespaces)
            desc = self._get_text(entry, ["description", "atom:summary", "atom:content"], namespaces)
            link = self._get_text(entry, ["link", "atom:link"], namespaces)
            pub_date = self._get_text(entry, ["pubDate", "atom:updated", "atom:published", "dc:date"], namespaces)

            if not title:
                continue

            # Clean HTML from description
            if desc:
                desc = self._strip_html(desc)[:300]

            item = NewsItem(
                title=title.strip(),
                source=feed_name.replace("_", " ").title(),
                category=feed_cfg["category"],
                url=link or "",
                description=desc or "",
                published=pub_date or "",
                priority=feed_cfg["priority"],
            )
            items.append(item)

        return items

    def _get_text(self, elem, tag_names: List[str], ns: Dict) -> str:
        """Try multiple tag names to extract text."""
        for tag in tag_names:
            if ":" in tag:
                prefix, name = tag.split(":", 1)
                found = elem.find(f"{prefix}:{name}", ns)
            else:
                found = elem.find(tag)
            if found is not None:
                # Handle Atom links (href attribute)
                if found.text:
                    return found.text
                elif found.get("href"):
                    return found.get("href")
        return ""

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    # ─── Finnhub News ───────────────────────────────────────────────────────

    def collect_finnhub_news(self) -> int:
        """Collect high-quality financial news from Finnhub."""
        if not self.cfg.FINNHUB_API_KEY:
            print("    [Finnhub] No API key — skipped")
            return 0

        print("  Collecting Finnhub news...")
        collected = 0

        try:
            # General market news
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general", "token": self.cfg.FINNHUB_API_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                for article in resp.json()[:20]:
                    item = NewsItem(
                        title=article.get("headline", ""),
                        source="Finnhub: " + article.get("source", ""),
                        category="macro",
                        url=article.get("url", ""),
                        description=article.get("summary", "")[:300],
                        published=datetime.fromtimestamp(article.get("datetime", 0)).isoformat() if article.get("datetime") else "",
                        priority=1,
                    )
                    if item.hash_id not in self._seen_hashes:
                        self._seen_hashes.add(item.hash_id)
                        self.items.append(item)
                        collected += 1

            print(f"    [Finnhub] {collected} items")
        except Exception as e:
            print(f"    [Finnhub] Error: {str(e)[:60]}")

        return collected

    # ─── Sentiment Scoring ──────────────────────────────────────────────────

    def score_sentiment(self):
        """Score sentiment for all collected items using keyword analysis."""
        print("\n  Scoring sentiment...")

        for item in self.items:
            text = f"{item.title} {item.description}".lower()

            bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
            bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

            total = bull_score + bear_score
            if total == 0:
                item.sentiment = "neutral"
                item.sentiment_score = 0.0
            else:
                score = (bull_score - bear_score) / total
                item.sentiment_score = round(score, 3)
                if score > 0.2:
                    item.sentiment = "bullish"
                elif score < -0.2:
                    item.sentiment = "bearish"
                else:
                    item.sentiment = "neutral"

            # Tag sectors
            item.sectors = []
            for sector, keywords in SECTOR_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    item.sectors.append(sector)

        # Summary
        sentiments = Counter(item.sentiment for item in self.items)
        print(f"    Bullish: {sentiments.get('bullish', 0)} | Bearish: {sentiments.get('bearish', 0)} | Neutral: {sentiments.get('neutral', 0)}")

    # ─── Earnings Calendar ──────────────────────────────────────────────────

    def collect_earnings_calendar(self) -> List[EarningsEvent]:
        """Fetch upcoming earnings from Finnhub."""
        if not self.cfg.FINNHUB_API_KEY:
            print("    [Earnings] No Finnhub key — skipped")
            return []

        print("  Collecting earnings calendar...")
        events = []

        try:
            today = datetime.now()
            from_date = today.strftime("%Y-%m-%d")
            to_date = (today + timedelta(days=14)).strftime("%Y-%m-%d")

            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"from": from_date, "to": to_date, "token": self.cfg.FINNHUB_API_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for e in data.get("earningsCalendar", [])[:50]:
                    events.append(EarningsEvent(
                        symbol=e.get("symbol", ""),
                        date=e.get("date", ""),
                        hour=e.get("hour", ""),
                        eps_estimate=e.get("epsEstimate") or 0.0,
                        eps_actual=e.get("epsActual") or 0.0,
                        revenue_estimate=e.get("revenueEstimate") or 0.0,
                    ))
            print(f"    [Earnings] {len(events)} upcoming events")
        except Exception as e:
            print(f"    [Earnings] Error: {str(e)[:60]}")

        return events

    # ─── Economic Calendar (FRED) ───────────────────────────────────────────

    def collect_economic_indicators(self) -> List[EconomicEvent]:
        """Fetch latest economic indicator values from FRED."""
        if not self.cfg.FRED_API_KEY:
            print("    [FRED] No API key — skipped")
            return []

        print("  Collecting economic indicators...")
        events = []

        for series_id, meta in FRED_SERIES.items():
            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self.cfg.FRED_API_KEY,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 2,
                    },
                    timeout=8,
                )
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    if len(obs) >= 1:
                        latest = float(obs[0].get("value", 0)) if obs[0].get("value", ".") != "." else 0
                        prev = float(obs[1].get("value", 0)) if len(obs) > 1 and obs[1].get("value", ".") != "." else latest
                        change = ((latest - prev) / prev * 100) if prev != 0 else 0

                        trend = "stable"
                        if abs(change) > 1:
                            trend = "rising" if change > 0 else "falling"

                        events.append(EconomicEvent(
                            name=meta["name"],
                            series_id=series_id,
                            category=meta["category"],
                            latest_value=latest,
                            previous_value=prev,
                            last_updated=obs[0].get("date", ""),
                            change_pct=round(change, 2),
                            trend=trend,
                        ))
            except Exception:
                pass

        print(f"    [FRED] {len(events)} indicators loaded")
        return events

    # ─── Sector Performance ─────────────────────────────────────────────────

    def collect_sector_performance(self) -> Dict:
        """Fetch sector performance from Alpha Vantage."""
        if not self.cfg.ALPHA_VANTAGE_KEY:
            return {}

        print("  Collecting sector performance...")
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "SECTOR", "apikey": self.cfg.ALPHA_VANTAGE_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = {}
                for timeframe in ["Rank A: Real-Time Performance", "Rank B: 1 Day Performance",
                                  "Rank C: 5 Day Performance", "Rank D: 1 Month Performance"]:
                    if timeframe in data:
                        result[timeframe] = {k: v for k, v in data[timeframe].items()}
                print(f"    [Sectors] {len(result)} timeframes loaded")
                return result
        except Exception as e:
            print(f"    [Sectors] Error: {str(e)[:60]}")
        return {}

    # ─── Save / Export ──────────────────────────────────────────────────────

    def save_all(self, earnings: List, economic: List, sectors: Dict):
        """Save all collected intelligence to data files."""
        timestamp = datetime.now().isoformat()

        # Market intelligence feed
        intel = {
            "timestamp": timestamp,
            "total_items": len(self.items),
            "sources": dict(Counter(item.source.split(":")[0] if ":" in item.source else item.source for item in self.items)),
            "items": [asdict(item) for item in sorted(self.items, key=lambda x: x.priority)],
        }
        save_json("market_intel.json", intel)

        # Sentiment summary
        sentiments = Counter(item.sentiment for item in self.items)
        sector_sentiment = {}
        for item in self.items:
            for sector in item.sectors:
                if sector not in sector_sentiment:
                    sector_sentiment[sector] = {"bullish": 0, "bearish": 0, "neutral": 0, "avg_score": []}
                sector_sentiment[sector][item.sentiment] += 1
                sector_sentiment[sector]["avg_score"].append(item.sentiment_score)

        for sector in sector_sentiment:
            scores = sector_sentiment[sector]["avg_score"]
            sector_sentiment[sector]["avg_score"] = round(sum(scores) / len(scores), 3) if scores else 0
            total = sector_sentiment[sector]["bullish"] + sector_sentiment[sector]["bearish"] + sector_sentiment[sector]["neutral"]
            sector_sentiment[sector]["bias"] = "bullish" if sector_sentiment[sector]["bullish"] > sector_sentiment[sector]["bearish"] else "bearish" if sector_sentiment[sector]["bearish"] > sector_sentiment[sector]["bullish"] else "neutral"

        avg_sentiment = sum(i.sentiment_score for i in self.items) / len(self.items) if self.items else 0

        sentiment_data = {
            "timestamp": timestamp,
            "overall": {
                "avg_score": round(avg_sentiment, 3),
                "bias": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
                "bullish_count": sentiments.get("bullish", 0),
                "bearish_count": sentiments.get("bearish", 0),
                "neutral_count": sentiments.get("neutral", 0),
            },
            "by_sector": sector_sentiment,
            "top_bullish": [asdict(i) for i in sorted(self.items, key=lambda x: x.sentiment_score, reverse=True)[:5]],
            "top_bearish": [asdict(i) for i in sorted(self.items, key=lambda x: x.sentiment_score)[:5]],
        }
        save_json("news_sentiment.json", sentiment_data)

        # Earnings
        if earnings:
            save_json("earnings_calendar.json", {
                "timestamp": timestamp,
                "count": len(earnings),
                "events": [asdict(e) for e in earnings],
            })

        # Economic
        if economic:
            save_json("economic_calendar.json", {
                "timestamp": timestamp,
                "count": len(economic),
                "indicators": [asdict(e) for e in economic],
            })

        # Sector performance
        if sectors:
            save_json("sector_performance.json", {
                "timestamp": timestamp,
                **sectors,
            })

        print(f"\n  Saved {len(self.items)} news items, {len(earnings)} earnings, {len(economic)} indicators")

    # ─── Generate AI-Consumable Context ─────────────────────────────────────

    def generate_intel_context(self) -> str:
        """Generate a text block for AI prompts summarizing current intelligence."""
        lines = ["MARKET INTELLIGENCE SUMMARY (auto-generated):"]

        # Overall sentiment
        sentiments = Counter(item.sentiment for item in self.items)
        avg = sum(i.sentiment_score for i in self.items) / len(self.items) if self.items else 0
        bias = "BULLISH" if avg > 0.1 else "BEARISH" if avg < -0.1 else "NEUTRAL"
        lines.append(f"  Overall sentiment: {bias} (score: {avg:+.3f})")
        lines.append(f"  Sources: {sentiments.get('bullish', 0)} bullish / {sentiments.get('bearish', 0)} bearish / {sentiments.get('neutral', 0)} neutral")

        # Top headlines by priority
        top = sorted(self.items, key=lambda x: (x.priority, -abs(x.sentiment_score)))[:8]
        lines.append("\n  TOP HEADLINES:")
        for item in top:
            icon = "▲" if item.sentiment == "bullish" else "▼" if item.sentiment == "bearish" else "─"
            lines.append(f"    {icon} [{item.source}] {item.title}")

        # Sector breakdown
        sector_counts = Counter()
        for item in self.items:
            for s in item.sectors:
                sector_counts[s] += 1
        if sector_counts:
            lines.append("\n  SECTOR COVERAGE:")
            for sector, count in sector_counts.most_common(8):
                lines.append(f"    {sector}: {count} mentions")

        context = "\n".join(lines)

        # Save for AI module consumption
        save_json("news_intel_context.json", {
            "timestamp": datetime.now().isoformat(),
            "context_text": context,
            "headline_count": len(self.items),
            "overall_sentiment": bias,
            "sentiment_score": round(avg, 3),
        })

        return context


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD DATA EXPORTER
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardExporter:
    """Export combined data for the web dashboard terminal."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.script_dir, "data")

    def _load(self, filename: str) -> Dict:
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def export_dashboard_data(self) -> Dict:
        """Combine all system data into a single dashboard export."""
        print_header("DASHBOARD DATA EXPORT")

        portfolio = self._load("portfolio_status.json")
        recommendations = self._load("recommendations.json")
        regime_ctx = self._load("regime_context.json")
        intel = self._load("market_intel.json")
        sentiment = self._load("news_sentiment.json")
        earnings = self._load("earnings_calendar.json")
        economic = self._load("economic_calendar.json")
        signal_perf = self._load("signal_performance.json")
        snapshots = self._load_snapshot_history()

        # Build account data
        acct = portfolio.get("account", {})
        account = {
            "equity": acct.get("equity", 0),
            "cash": acct.get("cash", 0),
            "buying_power": acct.get("buying_power", 0),
            "daily_pnl": acct.get("daily_pnl", 0),
            "daily_pnl_pct": acct.get("daily_pnl_pct", 0),
            "positions_count": len(portfolio.get("positions", [])),
        }

        # Build positions list
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

        # Build regime
        regime = {
            "name": regime_ctx.get("regime", "UNKNOWN").upper().replace("_", " "),
            "score": regime_ctx.get("regime_score", 0),
            "risk_appetite": regime_ctx.get("risk_appetite", 0.5),
            "stop_loss": regime_ctx.get("recommended_stop_loss_pct", 7.0),
            "take_profit": regime_ctx.get("recommended_take_profit_pct", 20.0),
            "cash_reserve": regime_ctx.get("recommended_cash_reserve_pct", 3.0),
            "max_positions": regime_ctx.get("recommended_max_positions", 20),
        }

        # Build signals
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

        # Build news feed
        news_items = []
        for item in (intel.get("items", []))[:20]:
            news_items.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "time": item.get("published", ""),
                "sentiment": item.get("sentiment", "neutral"),
                "category": item.get("category", ""),
            })

        # Build equity history from snapshots
        equity_history = []
        for snap in snapshots:
            equity_history.append({
                "date": snap.get("date", ""),
                "equity": snap.get("equity", 0),
                "benchmark": snap.get("benchmark", snap.get("equity", 0)),
            })

        # Protection summary
        protection_data = portfolio.get("protection_summary", {})
        protection = {
            "total": protection_data.get("total", len(positions)),
            "protected": protection_data.get("fully_protected", len(positions)),
            "trailing": protection_data.get("trailing_stops", 0),
            "degraded": protection_data.get("degraded", 0),
            "unprotected": protection_data.get("unprotected", 0),
            "fixed": protection_data.get("auto_fixed", 0),
        }

        export = {
            "export_timestamp": datetime.now().isoformat(),
            "account": account,
            "positions": positions,
            "regime": regime,
            "signals": signals,
            "news": news_items,
            "equity_history": equity_history,
            "tradeLog": [],  # TODO: pull from execution logs
            "protection": protection,
            "scoring": {"ai_confidence": 30, "momentum": 25, "technical": 20, "pnl": 15, "trend": 10},
            "sentiment_summary": sentiment.get("overall", {}),
            "earnings_upcoming": earnings.get("events", [])[:10],
            "economic_indicators": economic.get("indicators", []),
        }

        save_json("dashboard_export.json", export)
        print(f"  Dashboard data exported: {len(positions)} positions, {len(news_items)} news, {len(equity_history)} history points")
        return export

    def _load_snapshot_history(self) -> List[Dict]:
        """Load equity history from daily snapshots."""
        snapshots_dir = os.path.join(self.script_dir, "snapshots")
        history = []

        if not os.path.exists(snapshots_dir):
            return history

        for date_dir in sorted(os.listdir(snapshots_dir)):
            date_path = os.path.join(snapshots_dir, date_dir)
            if not os.path.isdir(date_path):
                continue

            # Look for post_rebalance snapshot first, then pre
            for prefix in ["post_rebalance", "pre_rebalance"]:
                for fname in os.listdir(date_path):
                    if fname.startswith(prefix) and fname.endswith(".json"):
                        try:
                            with open(os.path.join(date_path, fname), "r") as f:
                                snap = json.load(f)
                            acct = snap.get("account", {})
                            if acct.get("equity"):
                                history.append({
                                    "date": date_dir,
                                    "equity": float(acct["equity"]),
                                    "cash": float(acct.get("cash", 0)),
                                    "positions": len(snap.get("positions", {})),
                                })
                                break
                        except Exception:
                            pass
                else:
                    continue
                break

        return history


# ═══════════════════════════════════════════════════════════════════════════════
# DAEMON MODE (Scheduled Collection)
# ═══════════════════════════════════════════════════════════════════════════════

def run_daemon(interval_minutes: int = 30):
    """Run news collection on a schedule."""
    print(f"\n  Starting news daemon (every {interval_minutes} minutes)...")
    print(f"  Press Ctrl+C to stop\n")

    while True:
        try:
            run_full_collection()
            print(f"\n  Next collection at {(datetime.now() + timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n  Daemon stopped.")
            break
        except Exception as e:
            print(f"  Daemon error: {e}")
            time.sleep(60)


def run_full_collection():
    """Run complete news collection + export pipeline."""
    print_header("NEWS & MARKET INTELLIGENCE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    agg = NewsAggregator()

    # Collect all sources
    rss_count = agg.collect_rss_feeds()
    finnhub_count = agg.collect_finnhub_news()

    # Score sentiment
    agg.score_sentiment()

    # Earnings & economic
    earnings = agg.collect_earnings_calendar()
    economic = agg.collect_economic_indicators()
    sectors = agg.collect_sector_performance()

    # Save everything
    agg.save_all(earnings, economic, sectors)

    # Generate AI context
    context = agg.generate_intel_context()

    # Export dashboard data
    exporter = DashboardExporter()
    exporter.export_dashboard_data()

    print(f"\n  Collection complete: {rss_count + finnhub_count} news items")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Luxverum Capital - News & Market Intelligence")
    parser.add_argument("--collect", action="store_true", help="Collect from all sources")
    parser.add_argument("--sentiment", action="store_true", help="Score sentiment on collected news")
    parser.add_argument("--earnings", action="store_true", help="Refresh earnings calendar")
    parser.add_argument("--economic", action="store_true", help="Refresh economic indicators")
    parser.add_argument("--export", action="store_true", help="Export dashboard data")
    parser.add_argument("--full", action="store_true", help="Run complete pipeline")
    parser.add_argument("--daemon", action="store_true", help="Run on schedule")
    parser.add_argument("--interval", type=int, default=30, help="Daemon interval in minutes")

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.interval)
        return 0

    if args.full:
        return run_full_collection()

    agg = NewsAggregator()

    if args.collect:
        agg.collect_rss_feeds()
        agg.collect_finnhub_news()
        agg.score_sentiment()
        agg.save_all([], [], {})

    if args.earnings:
        earnings = agg.collect_earnings_calendar()
        save_json("earnings_calendar.json", {
            "timestamp": datetime.now().isoformat(),
            "count": len(earnings),
            "events": [asdict(e) for e in earnings],
        })

    if args.economic:
        economic = agg.collect_economic_indicators()
        save_json("economic_calendar.json", {
            "timestamp": datetime.now().isoformat(),
            "count": len(economic),
            "indicators": [asdict(e) for e in economic],
        })

    if args.export:
        exporter = DashboardExporter()
        exporter.export_dashboard_data()

    if not any([args.collect, args.earnings, args.economic, args.export, args.full]):
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
