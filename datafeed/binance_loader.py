import pandas as pd
from binance.client import Client


def load_binance(symbol="BTCUSDT", interval="1m", start=None, end=None, api_key=None, api_secret=None, limit=1000):
    """
    Loads OHLCV bars from Binance and returns DataFrame with
    date, open, high, low, close, volume
    - symbol: "BTCUSDT"
    - interval: "1m", "3m", "5m", "15m", "1h", "1d"
    - start, end: "2023-01-01", "2023-02-01", etc
    - limit: max bars per fetch (default 1000, max for most endpoints)
    """
    if api_key and api_secret:
        client = Client(api_key, api_secret)
    else:
        client = Client()
    # Binance intervals must be strings like '1m', '3m', '5m', '1h', '1d', etc.
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_str=start if start else "1 Jan, 2023",
        end_str=end if end else None,
        limit=limit
    )
    if not klines or len(klines) == 0:
        raise ValueError("No data returned from Binance.")

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    # Convert to expected columns
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df
