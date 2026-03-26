# XAUUSD Ultimate Pro AI Scalper

Institutional-grade trading bot for XAUUSD (Gold) using a hybrid Deep Learning architecture (CNN + LSTM + Self-Attention).

## Features
- **CNN-LSTM-Attention**: Advanced pattern recognition through historical candle sequences.
- **Bootstrap Caching**: Instant startup by loading recent data from local SQLite.
- **Multi-Account Support**: Seamlessly switch between Demo and Live accounts via `.env`.
- **Valetax Optimized**: Auto-detects specific symbol naming (e.g., `XAUUSD.vxc`).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your credentials in `.env` (use `.env.example` as a template).
3. Run the bot: `python xauusd_pro_bot.py`

*Note: Use at your own risk. Past performance does not guarantee future results.*
