import os
import threading
import schedule
import time
import logging
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify
from alpaca_trade_api import REST
from dotenv import load_dotenv

from bot.strategies import MLStrategy
from bot.risk_engine import RiskEngine
from bot.deep_rl_agent import DeepRLAgent

load_dotenv()


class EnhancedTradingBot:
    def __init__(self):
        # ================= ALPACA =================
        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")
        self.base_url = "https://paper-api.alpaca.markets"

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing Alpaca credentials in .env")

        self.api = REST(self.api_key, self.api_secret, self.base_url)

        # ================= SETTINGS =================
        self.auto_trading = True
        self.trade_history = []

        self.max_positions = 3
        self.cooldown_minutes = 15
        self.last_buy_time = None

        # STOCKS + CRYPTO (crypto = long only)
        self.auto_symbols = [
            "AAPL", "MSFT", "TSLA", "NVDA", "SPY",
            "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LTCUSD"
        ]

        # LONG RULES
        self.profit_target_long = 0.03      # +3%
        self.stop_loss_long = 0.02          # -2%

        # SHORT RULES (Option B)
        self.profit_target_short = 0.04     # +4%
        self.stop_loss_short = 0.012        # -1.2%

        # ================= ML + RISK + RL =================
        self.ml = MLStrategy()
        self.risk = RiskEngine()
        self.rl = RLAgent(aggressive=True)

        # ================= DATABASE =================
        self.setup_db()

        # ================= WEB SERVER =================
        self.app = Flask(__name__)
        self.setup_routes()

        # ================= LOGGING =================
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TradingBot")

        # DAILY ML RETRAIN
        schedule.every().day.at("04:30").do(self.daily_retrain)

    # ================= DATABASE =================
    def setup_db(self):
        self.conn = sqlite3.connect("trading_bot.db", check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                pnl REAL,
                timestamp TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equity REAL,
                timestamp TEXT
            )
        """)

        self.conn.commit()

    def log_trade(self, trade):
        self.cursor.execute("""
            INSERT OR REPLACE INTO trades (id, symbol, side, qty, price, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get("id"),
            trade.get("symbol"),
            trade.get("side"),
            trade.get("qty"),
            trade.get("price"),
            trade.get("pnl", 0),
            trade.get("timestamp")
        ))
        self.conn.commit()

    def log_equity(self):
        try:
            equity = float(self.api.get_account().equity)
            self.cursor.execute("""
                INSERT INTO equity_history (equity, timestamp)
                VALUES (?, ?)
            """, (equity, datetime.now().isoformat()))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Equity logging failed: {e}")

    # ================= DAILY ML RETRAIN =================
    def daily_retrain(self):
        self.logger.warning("🔁 DAILY ML RETRAIN STARTED")
        for symbol in self.auto_symbols:
            try:
                df = yf.Ticker(symbol).history(period="180d", interval="1h")
                if not df.empty:
                    self.ml.train(df)
                    self.logger.info(f"ML retrained on {symbol}")
            except Exception as e:
                self.logger.error(f"Retrain failed for {symbol}: {e}")
        self.logger.warning("✅ DAILY ML RETRAIN COMPLETE")

    # ================= WEB ROUTES =================
    def setup_routes(self):
        @self.app.route("/")
        def dashboard():
            return "✅ Trading Bot Running with ML + RL + SHORT SELLING"

        @self.app.route("/api/status")
        def status():
            positions = []
            for p in self.api.list_positions():
                positions.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "side": p.side
                })
            return jsonify({"positions": positions, "trades": self.trade_history[-20:]})

        @self.app.route("/api/equity")
        def equity():
            self.cursor.execute("""
                SELECT equity, timestamp FROM equity_history
                ORDER BY id ASC LIMIT 200
            """)
            rows = self.cursor.fetchall()
            return jsonify([{"equity": r[0], "timestamp": r[1]} for r in rows])

    # ================= BUY / SELL DECISIONS =================
    def scan_market_for_buys(self):
        if not self.auto_trading:
            return

        # Cooldown
        if self.last_buy_time:
            if (datetime.now() - self.last_buy_time).total_seconds() / 60 < self.cooldown_minutes:
                return

        # Position limit
        open_positions_api = self.api.list_positions()
        if len(open_positions_api) >= self.max_positions:
            return

        account_equity = float(self.api.get_account().equity)

        # Convert positions to risk-friendly format
        open_positions = [
            {"symbol": p.symbol, "qty": float(p.qty), "price": float(p.avg_entry_price), "side": p.side}
            for p in open_positions_api
        ]

        # Loop symbols
        for symbol in self.auto_symbols:
            try:
                df = yf.Ticker(symbol).history(period="90d", interval="30m")
                if df.empty:
                    continue

                if not self.ml.train(df):
                    continue

                signal, confidence = self.ml.predict_with_confidence(df)

                # RL modifies confidence + risk sizing
                rl_action = self.rl.get_action()
                confidence = min(1.0, confidence + rl_action["confidence_boost"])

                if rl_action["skip_trading"]:
                    return

                price = df["Close"].iloc[-1]
                atr = df["Close"].rolling(14).std().iloc[-1] * 2
                qty = self.risk.volatility_adjusted_size(price, atr, account_equity, confidence)
                qty = max(1, int(qty * rl_action["risk_multiplier"]))

                # ============= LONG ENTRY =============
                if signal == 1:
                    side = "buy"
                    stop_loss = self.stop_loss_long
                    profit_target = self.profit_target_long

                # ============= SHORT ENTRY =============
                elif signal == -1 and "USD" not in symbol:  # no crypto shorts
                    side = "sell"
                    stop_loss = self.stop_loss_short
                    profit_target = self.profit_target_short
                else:
                    continue

                # Execute order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc"
                )

                trade_data = {
                    "id": order.id,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "timestamp": datetime.now().isoformat()
                }

                self.trade_history.append(trade_data)
                self.log_trade(trade_data)
                self.last_buy_time = datetime.now()

                self.logger.info(f"✅ {side.upper()} {symbol} | Qty {qty} | Price {price:.2f}")
                return

            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")

    # ================= AUTO EXIT (LONG + SHORT) =================
    def auto_trading_cycle(self):
        self.scan_market_for_buys()

        for position in self.api.list_positions():
            symbol = position.symbol
            qty = float(position.qty)
            avg = float(position.avg_entry_price)
            current = float(position.current_price)
            side = position.side

            if side == "long":
                pct = (current - avg) / avg
                tp = self.profit_target_long
                sl = -self.stop_loss_long

            else:  # SHORT
                pct = (avg - current) / avg
                tp = self.profit_target_short
                sl = -self.stop_loss_short

            if pct >= tp or pct <= sl:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy" if side == "short" else "sell",
                    type="market",
                    time_in_force="gtc"
                )

                pnl = (current - avg) * qty * (-1 if side == "short" else 1)
                self.rl.update_after_trade(pnl)

                self.logger.info(f"🔻 CLOSED {symbol} ({side}) | PNL {pnl:.2f}")

    # ================= RUNNERS =================
    def run_trading_loop(self):
        schedule.every(5).minutes.do(self.auto_trading_cycle)
        schedule.every(1).minute.do(self.log_equity)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def run_web_server(self):
        self.app.run(host="0.0.0.0", port=5000, threaded=True)

    def run(self):
        threading.Thread(target=self.run_web_server, daemon=True).start()
        self.run_trading_loop()
