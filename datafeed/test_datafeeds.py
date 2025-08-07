# key, secret, url = "PK99OKUJAAI9KIINMT9G", "isFedDOxxFjl5KWuzSnPoe8Zb7aHMUYig4VDsuyI", "https://paper-api.alpaca.markets"
from datafeed import (
    load_csv, load_yfinance, load_alpaca,
    load_tvdatafeed, get_tv_ta, load_ccxt
)


def preview_df(df, feed_name):
    print(f"\n=== {feed_name} FEED PREVIEW (Last 10 Rows, Sorted by Date) ===")
    if df is None or df.empty:
        print("No data returned.")
        return
    print("Columns:", list(df.columns))
    df_sorted = df.sort_values(by='date')
    print(df_sorted.tail(10))
    print("=" * 50)


# Test CSV Loader (requires a sample CSV file)
try:
    csv_test_file = "test_data/BTCUSDT_3m.csv"  # Update this to any available CSV
    df_csv = load_csv(csv_test_file)
    preview_df(df_csv, "CSV")
except Exception as e:
    print("CSV feed error:", e)

# Test YFinance Loader
try:
    df_yf = load_yfinance("AAPL", "1d", "2023-01-01", "2030-02-01")
    preview_df(df_yf, "YFinance")
except Exception as e:
    print("YFinance feed error:", e)

# Test Alpaca Loader
try:
    # Add your Alpaca API credentials or load from env/secrets
    key, secret, url = "PK99OKUJAAI9KIINMT9G", "isFedDOxxFjl5KWuzSnPoe8Zb7aHMUYig4VDsuyI", "https://paper-api.alpaca.markets"
    df_alpaca = load_alpaca("BTC/USD", "1m", "2025-07-01", "2030-02-01", key, secret, url)
    preview_df(df_alpaca, "Alpaca")
except Exception as e:
    print("Alpaca feed error:", e)

# Test TVDatafeed Loader
try:
    df_tv = load_tvdatafeed("RGTI", "NASDAQ", "3m", 100)
    preview_df(df_tv, "TVDatafeed")
except Exception as e:
    print("TVDatafeed error:", e)

# Test TradingView_TA Loader
try:
    df_ta = get_tv_ta("AAPL", exchange="NASDAQ", interval="1d")
    preview_df(df_ta, "TradingView_TA")
except Exception as e:
    print("TradingView_TA error:", e)

# Test CCXT Loader (crypto)
try:
    df_ccxt = load_ccxt(symbol="BTC/USDT", timeframe="1m", limit=100, exchange="binanceus")
    preview_df(df_ccxt, "CCXT")
except Exception as e:
    print("CCXT feed error:", e)
