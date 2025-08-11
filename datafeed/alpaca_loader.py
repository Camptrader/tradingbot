from datetime import datetime, date, time, timedelta
import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def map_tf_to_alpaca(tf_str):
    tf_map = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "3m": TimeFrame(3, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }
    tf_str = str(tf_str).lower()
    if tf_str not in tf_map:
        raise ValueError(f"Unsupported timeframe for Alpaca: {tf_str}")
    return tf_map[tf_str]


def get_safe_end_time(user_end):
    """Return the effective end time excluding last 15 min, handling date or datetime input."""

    utc_now = datetime.now(pytz.UTC)
    cutoff_time = utc_now - timedelta(minutes=15)

    # Convert date-only to datetime at midnight UTC
    if isinstance(user_end, date) and not isinstance(user_end, datetime):
        user_end_dt = datetime.combine(user_end, time.min).replace(tzinfo=pytz.UTC)
    elif isinstance(user_end, datetime):
        user_end_dt = user_end
        if user_end_dt.tzinfo is None:
            user_end_dt = pytz.UTC.localize(user_end_dt)
    elif isinstance(user_end, str):
        # Parse string to datetime with UTC
        user_end_dt = pd.Timestamp(user_end)
        if user_end_dt.tzinfo is None:
            user_end_dt = user_end_dt.tz_localize('UTC')
        user_end_dt = user_end_dt.to_pydatetime()
    else:
        raise ValueError("Unsupported type for 'end' parameter")

    # Return earlier of user_end_dt and cutoff_time
    return min(user_end_dt, cutoff_time)


def load_alpaca(symbol, timeframe, start, end, key=None, secret=None, url=None, preview=True):
    start_dt = pd.Timestamp(start)
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize('UTC')

    # Always compute a UTC end
    end_dt_raw = pd.Timestamp(end)
    if end_dt_raw.tzinfo is None:
        end_dt_raw = end_dt_raw.tz_localize('UTC')

    # Stocks: exclude last 15 minutes; Crypto: keep raw end
    stock_end_dt  = get_safe_end_time(end_dt_raw)
    crypto_end_dt = end_dt_raw

    if '/' not in symbol:
        # STOCKS
        client = StockHistoricalDataClient(key, secret)
        tf = map_tf_to_alpaca(timeframe)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=map_tf_to_alpaca(timeframe),
            start=start_dt,
            end=stock_end_dt
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            raise ValueError("No bars returned from Alpaca (stock).")
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
        df = df.reset_index()
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], utc=True)
        else:
            raise ValueError("No 'timestamp' or 'time' column in bars dataframe.")
    else:
        # CRYPTO
        client = CryptoHistoricalDataClient()
        tf = map_tf_to_alpaca(timeframe)
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=map_tf_to_alpaca(timeframe),
            start=start_dt,
            end=crypto_end_dt
        )
        bars = client.get_crypto_bars(req)
        df = bars.df
        if df.empty:
            raise ValueError("No bars returned from Alpaca (crypto).")
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
        df = df.reset_index()
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], utc=True)
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], utc=True)
        else:
            raise ValueError("No 'timestamp' or 'time' column in bars dataframe.")

    df = df.rename(columns=str.lower)
    keep_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df = df[keep_cols]

    if preview:
        print(f"=== Alpaca FEED PREVIEW ({symbol}) ===")
        print("Columns:", list(df.columns))
        print(df.tail(10))

    return df
