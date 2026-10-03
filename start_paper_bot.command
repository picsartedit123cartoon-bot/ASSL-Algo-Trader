#!/bin/bash
cd /Users/siva/Downloads/angelone_chart_analysis_bot
source /Users/siva/angel-trading-bot/.venv/bin/activate

until curl -s --max-time 5 https://apiconnect.angelone.in >/dev/null; do
    sleep 10
done

caffeinate -dimsu python bot_paper_strategy.py
