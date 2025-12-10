import yfinance as yf
import pandas as pd
from bot.strategies import MLStrategy

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]
START_BALANCE = 10000
MAX_TOTAL_RISK_PCT = 0.06
DAILY_LOSS_LIMIT_PCT = 0.03
ATR_PERIOD = 14
TRAIN_WINDOW = 200


def get_atr_from_data(df, period=ATR_PERIOD):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def market_is_bullish_from_data(spy_df, idx):
    if idx < 200:
        return False
    window = spy_df.iloc[: idx + 1]
    sma50 = window["Close"].rolling(50).mean().iloc[-1]
    sma200 = window["Close"].rolling(200).mean().iloc[-1]
    price = window["Close"].iloc[-1]
    return price > sma50 > sma200


def backtest():
    balance = START_BALANCE
    start_of_day_equity = START_BALANCE

    position_open = False
    position_symbol = None
    position_qty = 0
    position_entry = 0
    position_stop = 0
    position_target = 0

    wins = 0
    losses = 0
    win_pnls = []
    loss_pnls = []

    print("Downloading data...")
    data = {}
    for sym in SYMBOLS:
        df = yf.Ticker(sym).history(period="2y", interval="1d")
        if df.empty:
            print(f"⚠ No data for {sym}, skipping.")
            continue
        df["ATR"] = get_atr_from_data(df)
        data[sym] = df

    if "SPY" not in data:
        print("❌ No SPY data — cannot run regime filter.")
        return

    spy_df = data["SPY"]
    ml_fast = MLStrategy()
    ml_slow = MLStrategy()

    spy_index = spy_df.index
    start_idx = max(TRAIN_WINDOW, 200, ATR_PERIOD)
    daily_trade_count = 0
    last_date = None

    for i in range(start_idx, len(spy_index) - 1):
        date = spy_index[i]
        next_date = spy_index[i + 1]

        if last_date is None or date.date() != last_date:
            daily_trade_count = 0
            start_of_day_equity = balance
            last_date = date.date()

        current_equity = balance
        daily_dd = (start_of_day_equity - current_equity) / start_of_day_equity
        if daily_dd >= DAILY_LOSS_LIMIT_PCT:
            print(f"🚨 Daily loss limit hit on {date.date()}")
            continue

        # Manage open position
        if position_open and position_symbol in data:
            sym_df = data[position_symbol]
            if next_date in sym_df.index:
                row = sym_df.loc[next_date]
                high = row["High"]
                low = row["Low"]

                closed = False
                pnl = 0

                if high >= position_target:
                    closed = True
                    pnl = (position_target - position_entry) * position_qty
                elif low <= position_stop:
                    closed = True
                    pnl = (position_stop - position_entry) * position_qty

                if closed:
                    balance += pnl
                    if pnl > 0:
                        wins += 1
                        win_pnls.append(pnl)
                    else:
                        losses += 1
                        loss_pnls.append(abs(pnl))

                    print(f"{next_date.date()} CLOSED {position_symbol} | PnL: {pnl:.2f}")
                    position_open = False

        if position_open:
            continue

        if not market_is_bullish_from_data(spy_df, i):
            continue

        print(f"\n{date.date()} — Checking trade conditions...")

        if daily_trade_count >= 3:
            continue

        for sym in SYMBOLS:
            if sym == "SPY" or sym not in data:
                continue

            sym_df = data[sym]
            if date not in sym_df.index:
                continue

            idx_sym = sym_df.index.get_loc(date)
            if idx_sym < TRAIN_WINDOW:
                continue

            atr = sym_df["ATR"].iloc[idx_sym]
            if pd.isna(atr):
                continue

            price = sym_df["Close"].iloc[idx_sym]
            vol_ratio = atr / price
            if vol_ratio < 0.004 or vol_ratio > 0.09:
                continue

            stop_loss_pct = (atr * 1.5) / price

            risk_amount = balance * 0.01
            position_value = risk_amount / stop_loss_pct
            qty = int(position_value / price)
            if qty < 1:
                continue

            open_risk = qty * price * stop_loss_pct
            if open_risk / balance > MAX_TOTAL_RISK_PCT:
                continue

            window = sym_df.iloc[idx_sym - TRAIN_WINDOW: idx_sym]

            trained_fast = ml_fast.train(window)
            trained_slow = ml_slow.train(window)
            if not trained_fast or not trained_slow:
                continue

            sig_fast, conf_fast = ml_fast.predict_with_confidence(window)
            sig_slow, conf_slow = ml_slow.predict_with_confidence(window)

            if conf_fast < 0.25 or conf_slow < 0.25:
                continue
            if not (sig_fast and sig_slow):
                continue

            confidence = 0.6 * conf_fast + 0.4 * conf_slow

            if confidence >= 0.80:
                risk_per_trade = 0.035
            elif confidence >= 0.60:
                risk_per_trade = 0.025
            elif confidence >= 0.50:
                risk_per_trade = 0.015
            else:
                risk_per_trade = 0.01

            risk_amount = balance * risk_per_trade
            position_value = risk_amount / stop_loss_pct
            qty = int(position_value / price)
            if qty < 1:
                continue

            stop_price = price * (1 - stop_loss_pct)
            tp_pct = (atr * 3.0) / price
            target_price = price * (1 + tp_pct)

            position_open = True
            position_symbol = sym
            position_qty = qty
            position_entry = price
            position_stop = stop_price
            position_target = target_price
            daily_trade_count += 1

            print(
                f"{date.date()} OPEN {sym} @ {price:.2f} | "
                f"Conf: {confidence:.2f} | Qty: {qty} | "
                f"Stop: {stop_price:.2f} | Target: {target_price:.2f}"
            )
            break

    print("\n===== ULTIMATE ENSEMBLE ML BACKTEST RESULTS =====")
    print(f"Starting Balance: {START_BALANCE:.2f}")
    print(f"Final Balance:    {balance:.2f}")
    print(f"Wins:   {wins}")
    print(f"Losses: {losses}")

    if wins + losses > 0:
        win_rate = wins / (wins + losses) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("Win Rate: No closed trades")

    if win_pnls and loss_pnls:
        avg_win = sum(win_pnls) / len(win_pnls)
        avg_loss = sum(loss_pnls) / len(loss_pnls)
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0
        expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)
        print(f"Average Win:  {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Expectancy:   {expectancy:.2f} per trade")


if __name__ == "__main__":
    backtest()