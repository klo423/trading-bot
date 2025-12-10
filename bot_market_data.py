# bot/market_data.py
import yfinance as yf
import asyncio
import threading
from alpaca_trade_api.stream import Stream
from bot import config

price_cache = {}  # latest prices

def handle_trade(trade):
    symbol = trade.symbol
    price_cache[symbol] = trade.price

def start_stream(symbols=None):
    symbols = symbols or config.DEFAULT_SYMBOLS
    stream = Stream(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.ALPACA_PAPER)

    for symbol in symbols:
        stream.subscribe_trades(handle_trade, symbol)

    thread = threading.Thread(target=asyncio.run, args=(stream._run_forever(),), daemon=True)
    thread.start()
def get_latest_price(symbol):
    """
    Fetch latest close price.
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching {symbol} price: {e}")
    return None
