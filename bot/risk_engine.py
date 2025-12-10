import math

class RiskEngine:
    """
    Institutional-grade risk engine:
    - Volatility adjusted sizing
    - Portfolio risk cap
    - Sector exposure cap
    - Correlation filter (basic version)
    """

    def __init__(self):
        # Max % of account at risk per trade (default 1%)
        self.base_risk_pct = 0.01

        # Max total portfolio risk at once (default 5%)
        self.max_portfolio_risk_pct = 0.05

        # Max % of account allocated per sector
        self.max_sector_pct = 0.30

        # Sector mapping (basic)
        self.sectors = {
            "AAPL": "TECH",
            "MSFT": "TECH",
            "NVDA": "TECH",
            "TSLA": "AUTO",
            "SPY": "INDEX"
        }

    # ✅ VOLATILITY-ADJUSTED POSITION SIZE
    def volatility_adjusted_size(self, price, atr, balance, confidence):
        """
        Uses:
        - Account balance
        - Volatility (ATR)
        - ML confidence boost
        """

        # Confidence boosts risk slightly
        if confidence >= 0.75:
            risk_pct = 0.02
        elif confidence >= 0.60:
            risk_pct = 0.015
        else:
            risk_pct = self.base_risk_pct

        dollar_risk = balance * risk_pct

        # Stop distance based on ATR
        stop_distance = atr * 1.5

        if stop_distance <= 0:
            return 1

        qty = dollar_risk / stop_distance
        return max(1, int(qty))

    # ✅ PORTFOLIO RISK CHECK
    def portfolio_risk_ok(self, open_positions, new_risk, balance):
        """
        Ensures total open risk never exceeds cap.
        """
        total_open_risk = sum(p["risk"] for p in open_positions)
        projected_risk = total_open_risk + new_risk

        return projected_risk / balance <= self.max_portfolio_risk_pct

    # ✅ SECTOR EXPOSURE CHECK
    def sector_exposure_ok(self, open_positions, symbol, qty, price, balance):
        """
        Limits over-exposure to a single sector.
        """
        sector = self.sectors.get(symbol, "OTHER")
        sector_exposure = 0

        for p in open_positions:
            sym_sector = self.sectors.get(p["symbol"], "OTHER")
            if sym_sector == sector:
                sector_exposure += p["qty"] * p["price"]

        new_exposure = qty * price
        projected = sector_exposure + new_exposure

        return projected / balance <= self.max_sector_pct

    # ✅ CORRELATION FILTER (BASIC VERSION)
    def correlation_ok(self, symbol, open_positions):
        """
        Prevents stacking highly correlated names.
        (Basic rule: no duplicate symbols)
        """
        for p in open_positions:
            if p["symbol"] == symbol:
                return False
        return True
