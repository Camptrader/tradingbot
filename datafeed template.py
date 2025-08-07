# =======================================
# SETUP: Install requirements (Uncomment if needed)
# =======================================
# !pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
# !pip install tradingview_ta ta matplotlib backtrader pandas

import backtrader as bt
import matplotlib.pyplot as plt
# =======================================
# 1. Imports and Configuration
# =======================================
import pandas as pd
import ta
from tradingview_ta import TA_Handler, Interval as TA_Interval
from tvDatafeed import TvDatafeedLive, Interval

# ---- CONFIG ----
symbol = "NVDA"
exchange = "NASDAQ"
tv_interval = Interval.in_15_minute
bars_to_fetch = 500

# =======================================
# 3. Historical Data Fetch (tvdatafeed)
# =======================================
tvl = TvDatafeedLive()  # No username/password, uses auth token from env
df = tvl.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=bars_to_fetch)
df = df.reset_index()  # Ensure DataFrame has integer index
print(df.tail(3))
df.columns = [col.strip().lower() for col in df.columns]
print(df.columns)  # Debug: should be Index(['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'], ...)

# =======================================
# 3. Local Technical Indicators (ta)
# =======================================
df = pd.DataFrame({'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
df['ema3'] = ta.trend.EMAIndicator(close=df['close'], window=3).ema_indicator()
print(df)
df['ema20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
macd_obj = ta.trend.MACD(close=df['close'])
df['macd'] = macd_obj.macd()
df['macd_signal'] = macd_obj.macd_signal()

print(df[['close', 'ema20', 'rsi', 'macd', 'macd_signal']].tail(5))

# =======================================
# 4. TradingView TA Snapshot (tradingview_ta)
# =======================================
handler = TA_Handler(
    symbol=symbol,
    screener="america",
    exchange=exchange,
    interval=TA_Interval.INTERVAL_15_MINUTES,
)
ta_summary = handler.get_analysis()
print("TradingView Recommendation:", ta_summary.summary)
print("TradingView Indicators:", ta_summary.indicators)

# =======================================
# 5. Matplotlib Visualization
# =======================================
plt.figure(figsize=(14, 6))
plt.plot(df['close'], label='Close Price')
plt.plot(df['ema20'], label='EMA 20')
plt.title(f"{symbol} - 15m Price and EMA")
plt.legend()
plt.show()

plt.figure(figsize=(14, 3))
plt.plot(df['macd'], label='MACD')
plt.plot(df['macd_signal'], label='MACD Signal')
plt.title("MACD")
plt.legend()
plt.show()


# =======================================
# 6. (Optional) Run Simple Backtrader Strategy
# =======================================
class EMARSI(bt.Strategy):
    params = dict(ema_period=20, rsi_period=14, rsi_low=30, rsi_high=70)

    def __init__(self):
        self.ema = bt.ind.EMA(period=self.p.ema_period)
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.ema[0] and self.rsi[0] < self.p.rsi_low:
                self.buy()
        else:
            if self.rsi[0] > self.p.rsi_high:
                self.sell()


# Convert DataFrame for Backtrader
bt_df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
bt_df['datetime'] = pd.to_datetime(bt_df['datetime'])
bt_df.set_index('datetime', inplace=True)

data = bt.feeds.PandasData(dataname=bt_df)
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(EMARSI)
results = cerebro.run()
cerebro.plot()


# =======================================
# 7. Live Bar Callback/Streaming (TvDatafeedLive)
# =======================================
def on_new_bar(bar):
    print("New bar received:", bar)
    # Optionally, recalc indicators, plot, or run live signal logic here


seis = tvl.new_seis(symbol=symbol, exchange=exchange, timeframe=tv_interval)
seis.add_consumer(on_new_bar)
print("Starting live streaming... (Ctrl+C to stop)")
try:
    tvl.run()
except KeyboardInterrupt:
    print("Stopped live streaming.")
