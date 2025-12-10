import yfinance as yf
data = yf.Ticker("AAPL").history(period="1y")
print(data.tail())
