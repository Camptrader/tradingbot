import ccxt
import pandas as pd


def load_ccxt(symbol="BTC/USDT", timeframe="1m", since=None, limit=1000, exchange="binanceus", preview=True):
    ex = getattr(ccxt, exchange)()
    if since is not None and isinstance(since, str):
        since_ms = int(pd.Timestamp(since).timestamp() * 1000)
    else:
        since_ms = since

    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    if not bars or len(bars) == 0:
        raise ValueError(f"No data returned from {exchange} for {symbol} ({timeframe})")
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    df.columns = [c.lower() for c in df.columns]  # Make sure columns are lowercase

    if preview:
        print("\n=== CCXT FEED PREVIEW ===")
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
        print("=" * 30)

    return df
