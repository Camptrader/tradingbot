def load_yfinance(symbol, interval, start, end):
    import yfinance as yf
    import pandas as pd

    df = yf.download(symbol, interval=interval, start=start, end=end)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}) from yfinance.")

    df = df.reset_index()
    # Handle possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(level) for level in col if level]) for col in df.columns.values]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    # Find the datetime column (now could be 'date', 'datetime', 'index', or similar)
    date_col = None
    for candidate in ['date', 'datetime', 'index', 'date_time']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # often, the first column is the date after reset_index
        date_col = df.columns[0]
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[~df['date'].isna()]

    # Try to find OHLCV columns, robust to column names like "open", "open_x", "open_"
    def find_col(cols, base):
        for c in cols:
            if c.lower() == base or c.lower().startswith(base + "_"):
                return c
        raise ValueError(f"Missing '{base}' column in yfinance DataFrame.")

    open_col = find_col(df.columns, 'open')
    high_col = find_col(df.columns, 'high')
    low_col = find_col(df.columns, 'low')
    close_col = find_col(df.columns, 'close')
    volume_col = find_col(df.columns, 'volume')

    return df[['date', open_col, high_col, low_col, close_col, volume_col]].rename(
        columns={
            open_col: 'open',
            high_col: 'high',
            low_col: 'low',
            close_col: 'close',
            volume_col: 'volume'
        }
    )
