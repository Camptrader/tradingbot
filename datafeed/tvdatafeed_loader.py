def load_tvdatafeed(symbol, exchange, interval='3m', n_bars=2600, username=None, password=None,
                    local_tz='America/Los_Angeles'):
    from tvDatafeed import TvDatafeed, Interval
    import pandas as pd
    tv = TvDatafeed(username, password) if username and password else TvDatafeed()
    interval_map = {
        '1m': Interval.in_1_minute,
        '3m': Interval.in_3_minute,
        '5m': Interval.in_5_minute,
        '15m': Interval.in_15_minute,
        '30m': Interval.in_30_minute,
        '1h': Interval.in_1_hour,
        '4h': Interval.in_4_hour,
        '1d': Interval.in_daily
    }
    data = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval_map[interval], n_bars=n_bars)
    if data is None or data.empty:
        raise ValueError("No data returned from TradingView")
    df = data.reset_index()
    df.columns = [col.lower() for col in df.columns]

    # ---- Timezone Handling ----
    dt_col = 'datetime' if 'datetime' in df.columns else 'time'
    if local_tz:  # User specifies the local timezone (e.g., 'America/Los_Angeles' or 'Etc/GMT+7')
        df['date'] = (
            pd.to_datetime(df[dt_col], errors='coerce')
            .dt.tz_localize(local_tz)
            .dt.tz_convert('UTC')
        )
    else:  # Assume already in UTC (legacy behavior)
        df['date'] = pd.to_datetime(df[dt_col], utc=True, errors='coerce').dt.tz_convert(None)
    df = df[~df['date'].isna()]
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
