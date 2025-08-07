import yfinance as yf
import pandas as pd

# Choose a ticker (almost any will do, some keys may be missing for thin stocks)
t = yf.Ticker("AAPL")
fields = list(t.info.keys())

# Print them all
for f in fields:
    print(f)