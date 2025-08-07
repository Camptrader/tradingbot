import pandas as pd
import matplotlib.pyplot as plt

# -- File names, must be in project folder
ALPACA_PATH = "POET_3m_loaded_data.csv"
TV_PATH = "NASDAQ_POET, 3_2b4e5.csv"

# --- Load both datasets
alpaca = pd.read_csv(ALPACA_PATH)
tv = pd.read_csv(TV_PATH)

# --- Detect correct datetime column in TV file
tv_datetime_col = None
for col in ['date', 'time', 'Time', 'timestamp', 'Timestamp', 'datetime']:
    if col in tv.columns:
        tv_datetime_col = col
        break
if tv_datetime_col is None:
    raise ValueError(f"Could not find datetime column in TV file! Columns: {tv.columns.tolist()}")

alpaca['date'] = pd.to_datetime(alpaca['date'], utc=True, errors='coerce')
tv['date'] = pd.to_datetime(tv[tv_datetime_col], utc=True, errors='coerce')

# --- Align by date (inner join: only bars present in both)
merged = pd.merge(alpaca, tv, how='inner', on='date', suffixes=('_alpaca', '_tv'))

print(f"Bars in both: {len(merged)} (Alpaca: {len(alpaca)}, TV: {len(tv)})")

# --- Compare close prices
merged['close_diff'] = merged['close_alpaca'] - merged['close_tv']
max_diff = merged['close_diff'].abs().max()
print(f"Max difference in close: {max_diff}")

print("\nRows with |close_diff| > 0.01:")
print(merged.loc[merged['close_diff'].abs() > 0.01, ['date', 'close_alpaca', 'close_tv', 'close_diff']].head(10))

# --- Plot closes overlay
plt.figure(figsize=(16,6))
plt.plot(merged['date'], merged['close_alpaca'], label='Alpaca Close', color='blue')
plt.plot(merged['date'], merged['close_tv'], label='TV Close', color='red', alpha=0.6)
plt.title("Close Prices: Alpaca vs TradingView Export")
plt.legend()
plt.tight_layout()
plt.show()

# --- If EMAs or other indicators exist in TV export, compare those too
ema_cols = [col for col in tv.columns if "ema" in col.lower()]
for col in ema_cols:
    col_tv = col
    # Find matching EMA column in Alpaca, if present
    col_alpaca = col + "_alpaca" if (col + "_alpaca") in merged.columns else None
    if col_tv in merged.columns:
        plt.figure(figsize=(16,6))
        plt.plot(merged['date'], merged[col_tv], label=f'TV {col}', color='red', alpha=0.7)
        if col_alpaca and col_alpaca in merged.columns:
            plt.plot(merged['date'], merged[col_alpaca], label=f'Alpaca {col}', color='blue')
        plt.title(f"EMA Comparison: {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()
