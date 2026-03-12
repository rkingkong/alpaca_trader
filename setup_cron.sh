#!/bin/bash
# Luxverum Capital — Cron Setup
# Run this to set up scheduled news collection

TRADER_DIR="/opt/luxverum/trading"
PYTHON="/usr/bin/python3"

# Add cron jobs
(crontab -l 2>/dev/null; echo "# Luxverum Capital — News Collection (every 30 min during market hours)") | crontab -
(crontab -l 2>/dev/null; echo "*/30 6-17 * * 1-5 cd $TRADER_DIR && $PYTHON 11_news_aggregator.py --full >> data/news_cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "# Luxverum Capital — Full morning collection (6 AM weekdays)") | crontab -
(crontab -l 2>/dev/null; echo "0 6 * * 1-5 cd $TRADER_DIR && $PYTHON 11_news_aggregator.py --full >> data/news_cron.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "# Luxverum Capital — Dashboard export (every hour)") | crontab -
(crontab -l 2>/dev/null; echo "0 * * * * cd $TRADER_DIR && $PYTHON 11_news_aggregator.py --export >> data/export_cron.log 2>&1") | crontab -

echo "Cron jobs installed. Verify with: crontab -l"
