import pandas as pd
from tradingview_ta import TA_Handler, Interval


def get_tv_ta(symbol, screener='america', exchange='NASDAQ', interval='3m'):
    # Interval options: Interval.INTERVAL_1_MINUTE, etc.
    interval_map = {
        '1m': Interval.INTERVAL_1_MINUTE,
        '5m': Interval.INTERVAL_5_MINUTES,
        '15m': Interval.INTERVAL_15_MINUTES,
        '30m': Interval.INTERVAL_30_MINUTES,
        '1h': Interval.INTERVAL_1_HOUR,
        '4h': Interval.INTERVAL_4_HOURS,
        '1d': Interval.INTERVAL_1_DAY,
        '1w': Interval.INTERVAL_1_WEEK,
        '1M': Interval.INTERVAL_1_MONTH
    }
    handler = TA_Handler(
        symbol=symbol,
        screener="america",
        exchange=exchange,
        interval=interval_map[interval]
    )
    analysis = handler.get_analysis()
    indicators = analysis.indicators
    # For demo, turn into single-row DataFrame
    df = pd.DataFrame([indicators])
    return df
