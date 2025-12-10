# bot/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Required Alpaca credentials
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing Alpaca API credentials. Set ALPACA_KEY and ALPACA_SECRET.")

# Optional settings
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
DB_PATH = os.getenv("DB_PATH", "trading_bot.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Trading defaults
POSITION_SIZE = int(os.getenv("POSITION_SIZE", 10))
PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", 0.03))
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.02))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))
DEFAULT_SYMBOLS = os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,TSLA,SPY,QQQ,GLD,BTCUSD").split(",")

# Debug
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
if DEBUG:
    print(f"Config loaded. Symbols: {DEFAULT_SYMBOLS}, DB: {DB_PATH}")
