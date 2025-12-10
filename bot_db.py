# bot/db.py
import sqlite3
from datetime import datetime
from bot import config

class Database:
    def __init__(self, db_path=config.DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                timestamp TEXT,
                pnl REAL,
                auto BOOLEAN,
                bracket BOOLEAN
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                qty INTEGER,
                avg_price REAL,
                current_price REAL,
                unrealized_pnl REAL,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def log_trade(self, symbol, side, qty, price, pnl=0.0, auto=False, bracket=False):
        self.cursor.execute("""
            INSERT INTO trades (symbol, side, qty, price, timestamp, pnl, auto, bracket)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, qty, price, datetime.now().isoformat(), pnl, auto, bracket))
        self.conn.commit()

    def close(self):
        self.conn.close()
