from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from streamlit import header


def load_alpaca(symbol, tf, start, end, key, secret, url):
    api = REST(key, secret, url)
    tf_map = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "3m": TimeFrame(3, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }
    tf_obj = tf_map.get(tf)
    if tf_obj is None:
        raise ValueError(f"Unsupported timeframe: {tf}")
    # Use 'iex' feed to avoid SIP errors
    bars = api.get_bars(symbol, tf_obj, start=start, end=end, feed='iex').df
    if bars.empty:
        raise ValueError("No bars returned from Alpaca.")
    bars = bars.reset_index()
    bars.columns = [col.lower() for col in bars.columns]
    bars.rename(columns={"timestamp": "date"}, inplace=True)
    return bars[["date", "open", "high", "low", "close", "volume"]]


print(header)
