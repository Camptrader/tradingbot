Universal Backtester Onboarding Guide
Welcome to the Universal Backtester – a flexible framework for systematic trading research, rapid prototyping, and walk-forward optimization. This guide covers core data sources, built-in strategies, parameter spaces, and example workflows for fast onboarding.
________________________________________
Table of Contents
1.	Overview
2.	Supported Data Feeds
3.	Built-in Strategies
o	RMA Strategy
o	Crypto Intraday Multi-Signal
o	SMA Cross
o	Crypto Hybrid ATR Mom Break
4.	Parameter Spaces
5.	Typical Use-Cases
6.	Sample Workflow
7.	Extending the System
8.	Halal Compliance
9.	FAQ & Troubleshooting
________________________________________
Overview
The Universal Backtester supports multi-asset strategy research, optimization, and parameter sweeps across stocks and crypto. With built-in support for Halal (no short selling), session controls, and unified OHLCV data formatting, it's your all-in-one platform for strategy validation.
________________________________________
Supported Data Feeds
Feed Name	Loader Function	Typical Use-Case	Notes
CSV	load_csv	Custom/historical data	Quick prototyping, local data
Yahoo Finance	load_yfinance	US stocks, indices	Daily, 1min+, equities only
Alpaca	load_alpaca	Stocks/crypto (live/paper)	Requires API key
TradingView Data	load_tvdatafeed	All asset types, fast	No login = limited symbols
TradingView TA	get_tv_ta	Technical analysis snapshot	Single-bar indicators only
Binance	load_binance	Crypto, spot only	Direct, rate-limited
CCXT (multi-ex)	load_ccxt	Crypto, multiple exchanges	Good for broad asset coverage
Note: All loaders normalize to columns: date, open, high, low, close, volume.
________________________________________
Built-in Strategies
RMA Strategy
•	Type: Impulse/Trend-following, session aware
•	Assets: Stocks, Crypto
•	Halal: Long-only, no leverage
•	Best for: Range/Trend assets, intraday and swing
Crypto Intraday Multi-Signal
•	Type: Breakout + momentum + trend + volatility filters
•	Assets: Crypto
•	Halal: Long-only
•	Best for: Volatile, liquid crypto pairs
SMA Cross
•	Type: Classic Moving Average Crossover
•	Assets: Stocks, Crypto
•	Halal: Long-only
•	Best for: Trend capture, simple equities systems
Crypto Hybrid ATR Mom Break
•       Type: ATR + momentum breakout with partial exits
•       Assets: Crypto
•       Halal: Long-only
•       Best for: Breakout momentum with volatility filter
________________________________________
Parameter Spaces
Below are the main parameter sweeps as defined in the platform (CD_PARAM_SPACES). Most strategies can be fine-tuned for your asset and timeframe.
RMA Strategy
Parameter	Default	Range	Step	Notes
rma_len	50	40 to 100	5	RMA smoothing length
barsforentry	2	1 to 10	1	Min. consecutive green bars
barsforexit	2	1 to 10	1	Min. consecutive red bars to exit
atrlen	9	2 to 20	1	ATR length for normalization
normalizedupper	0.001	0 to 2	0.1	Impulse entry threshold (upper)
normalizedlower	-0.001	0 to -2	-0.1	Impulse entry threshold (lower)
ema_fast_len	5	2 to 30	1	Fast EMA for trend filter
ema_slow_len	60	3 to 100	1	Slow EMA for trend filter
emasrc	45	[15,30,45,60...]	fixed	Source EMA length
risklen	50	40 to 0	-0.5	Stop loss in %
trailpct	50	40 to 0	-0.5	Trailing stop in %
Session	"13:31"-"19:52"	-	-	Session start/end (HH:MM)
maxtradesperday	1	-	-	Maximum trades per day
Crypto Intraday Multi-Signal
Parameter	Default	Range	Step
breakout_len	20	5 to 51	5
momentum_len	10	5 to 51	5
momentum_thresh	0.5	0.5 to 3.0	0.3
trend_len	50	10 to 100	1
atr_len	14	2 to 20	1
min_atr	0.5	0.1 to 2.1	0.2
trailing_stop_pct	2.0	0.5 to 5.5	0.5
max_hold_bars	3000	5 to 50	1
maxtradesperday	20	-	-
SMA Cross
Parameter	Default	Range	Step
fast_len	5	2 to 16	2
slow_len	20	5 to 61	5
Crypto Hybrid ATR Mom Break
Parameter       Default Range   Step
breakout_len    20      10-60   5
ema_len         50      20-100  5
roc_thresh      0.5     0.1-2.0 0.1
atr_len         14      5-30    1
atr_mult        2.0     1.0-3.0 0.5
________________________________________
Typical Use-Cases
•	Strategy Optimization: Use the coordinate descent optimizer to tune any parameter set for maximum Sharpe, Sortino, or profit.
•	Walk-Forward Research: Export optimized parameter sets and apply walk-forward to test robustness on unseen data.
•	Symbol/Timeframe Portability: Swap data source (e.g., TradingView, Alpaca, CSV) and re-use the same strategy logic across stocks or crypto.
•	Rapid Strategy Prototyping: Add new functions and register them in STRATEGY_REGISTRY to test new alpha signals.
________________________________________
Sample Workflow
1.	Load Data
o	Upload a CSV or connect to an API (e.g., Yahoo, Alpaca, Binance).
o	Select symbol and timeframe.
2.	Choose Strategy
o	Pick from RMA, Intraday Multi-Signal, SMA Cross, or Hybrid ATR Mom Break.
o	Adjust parameter ranges as needed.
3.	Set Backtest Range
o	Use the date picker to filter the data for your test period.
4.	Run Backtest
o	Hit run and view the trade log, equity curve, and summary stats.
5.	Optimize Parameters
o	Use built-in optimizer for maximum return/risk-adjusted performance.
6.	Export Results
o	Download trade logs or parameter sets for further research or walk-forward analysis.
________________________________________
Extending the System
•	Add a New Data Feed:
Write a loader function (input: config, output: DataFrame w/ date, ohlcv) and register it in the UI.
•	Add a New Strategy:
Write a function: def my_strategy(df, param1, param2, ...), return trade log DataFrame. Register in STRATEGY_REGISTRY.
•	Custom Metrics:
Add custom performance metrics to the summary panel for deeper analytics.
________________________________________
Halal Compliance
•	All strategies are long-only (no short selling).
•	No margin or interest-based trading.
•	Trade caps and risk controls to avoid excessive speculation.
________________________________________
FAQ & Troubleshooting
•	"No data returned" error: Check API keys, symbol names, and network.
•	Session time filter not working: Make sure your date column is properly parsed to datetime.
•	Strategy not taking trades: Review entry parameter ranges or check if filters are too strict.
•	API rate limits: Use CSV or reduce frequency for high-volume data feeds.
________________________________________
For more info, see the code comments or reach out to the maintainer. Happy backtesting!
________________________________________

