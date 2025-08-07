# How to use             from .csv_loader import load_csv  # No changes needed if same format!
import pandas as pd


def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.lower() for c in df.columns]
    # Always create a 'date' column from possible time columns
    if 'date' not in df.columns:
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        else:
            raise ValueError("CSV must have a 'date', 'datetime', or 'time' column.")
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df['date'] = df['date'].dt.tz_convert(None)  # Always remove tz info for compatibility
    df = df[~df['date'].isna()]
    print("Columns:", df.columns.tolist())
    print("First 3 rows:\n", df.head(3))

    return df
