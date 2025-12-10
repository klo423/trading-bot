import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MLStrategy:
    """
    Enhanced Ensemble ML model:
    Models:
      - RandomForest
      - GradientBoosting
      - LogisticRegression

    Features:
      - return             (1-bar return)
      - rsi                (14-period RSI)
      - ema_trend          (EMA(8) > EMA(21) ?)
      - momentum           (5-bar return)
      - vol_spike          (volume / 20-day avg volume)
      - atr_vol            (ATR-normalized volatility)
      - bb_pos             (position inside Bollinger bands)
      - macd               (MACD line)
      - macd_signal_diff   (MACD - signal line)

    Output:
      - signal: 1 (buy) or 0 (no trade)
      - confidence: 0.0 – 1.0
    """

    def __init__(self):
        self.models = {
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            ),
            "lr": LogisticRegression(
                max_iter=4000,
                n_jobs=-1,
            ),
        }
        self.scaler = StandardScaler()
        self.trained = False

    # ========= Helper indicators =========
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

    def macd(self, close: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    # ========= Feature builder =========
    def prepare_data(self, df: pd.DataFrame):
        df = df.copy()

        # Basic guards
        needed_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not needed_cols.issubset(df.columns):
            return None, None

        # Core features
        df["return"] = df["Close"].pct_change()
        df["rsi"] = self.rsi(df["Close"], 14)

        df["ema8"] = df["Close"].ewm(span=8).mean()
        df["ema21"] = df["Close"].ewm(span=21).mean()
        df["ema_trend"] = (df["ema8"] > df["ema21"]).astype(int)

        df["momentum"] = df["Close"].pct_change(5)
        df["vol_spike"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # ATR-based volatility
        df["atr"] = self.atr(df, 14)
        df["atr_vol"] = df["atr"] / df["Close"]

        # Bollinger Bands position (%B)
        bb_mid = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_pos"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # MACD
        macd_line, signal_line, hist = self.macd(df["Close"])
        df["macd"] = macd_line
        df["macd_signal_diff"] = macd_line - signal_line

        # Drop NaNs from indicators
        df = df.dropna()

        # Need enough data for training
        if len(df) < 150:
            return None, None

        feature_cols = [
            "return",
            "rsi",
            "ema_trend",
            "momentum",
            "vol_spike",
            "atr_vol",
            "bb_pos",
            "macd",
            "macd_signal_diff",
        ]

        X = df[feature_cols]
        y = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # Align labels (drop last NaN)
        y = y.loc[X.index]
        y = y.dropna()
        X = X.loc[y.index]

        if len(X) == 0:
            return None, None

        return X, y

    # ========= TRAIN =========
    def train(self, data: pd.DataFrame) -> bool:
        X, y = self.prepare_data(data)
        if X is None or len(X) == 0:
            self.trained = False
            return False

        X_scaled = self.scaler.fit_transform(X)

        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
            except Exception:
                # If any model fails, keep going with others
                continue

        self.trained = True
        return True

    # ========= PREDICT WITH CONFIDENCE =========
    def predict_with_confidence(self, data: pd.DataFrame):
        """
        Returns:
          signal: 1 (buy) or 0 (no trade)
          confidence: float between 0 and 1
        """
        if not self.trained:
            return 0, 0.0

        X, _ = self.prepare_data(data)
        if X is None or len(X) == 0:
            return 0, 0.0

        X_scaled = self.scaler.transform(X)
        probs = []
        votes = []

        # Use only models that have predict_proba
        for name, model in self.models.items():
            try:
                p = model.predict_proba(X_scaled)[-1][1]
            except Exception:
                # Fallback to hard prediction if no probas
                pred = model.predict(X_scaled)[-1]
                p = 0.7 if pred == 1 else 0.3

            v = 1 if p >= 0.5 else 0
            probs.append(p)
            votes.append(v)

        if not probs:
            return 0, 0.0

        # Weighted ensemble confidence (RF & GB heavier)
        if len(probs) == 3:
            confidence = 0.4 * probs[0] + 0.4 * probs[1] + 0.2 * probs[2]
        else:
            confidence = float(np.mean(probs))

        # Majority vote for direction
        signal = 1 if sum(votes) >= (len(votes) // 2 + 1) else 0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return signal, confidence
