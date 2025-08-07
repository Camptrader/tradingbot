# ===========================================
# 1. IMPORTS AND UTILITY FUNCTIONS
# ===========================================
# Import core libraries, trading/datafeed modules, and parameter registry helpers.

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import os
from datafeed import load_csv, load_yfinance, load_alpaca, load_tvdatafeed, get_tv_ta, load_ccxt
from strategies import crypto_intraday_multi
from strategies.rma import rma_strategy
from strategies.sma_cross import sma_cross_strategy

# ===========================================
# 2. PARAMETER REGISTRY (SAVE/LOAD TO JSON)
# ===========================================
# These helpers persist and retrieve best parameters/results for each strategy/symbol pair.

PARAM_REGISTRY_FILE = "best_params.json"
def load_param_registry():
    if os.path.exists(PARAM_REGISTRY_FILE):
        with open(PARAM_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_param_registry(registry):
    with open(PARAM_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def get_saved_params(strategy, symbol):
    registry = load_param_registry()
    return registry.get(strategy, {}).get(symbol, None)

def register_params(strategy, symbol, params, results=None):
    registry = load_param_registry()
    if strategy not in registry:
        registry[strategy] = {}
    registry[strategy][symbol] = {"params": params, "results": results}
    save_param_registry(registry)

# ===========================================
# 3. STRATEGY DEFINITION
# ===========================================
# This dictionary defines available strategies, their params, and universal keys.

STRATEGY_REGISTRY = {
    "RMA Strategy": {
        "function": rma_strategy,
        "params": [
            "rma_len", "barsforentry", "barsforexit", "atrlen",
            "normalizedupper", "normalizedlower", "ema_fast_len", "ema_slow_len",
            "emasrc", "risklen", "trailpct"
        ],
        "universal": ["session_start", "session_end", "keeplime", "initial_capital", "qty", "maxtradesperday",
                      "use_session_end_rule"]
    },
    "Crypto Intraday Multi-Signal": {
        "function": crypto_intraday_multi,
        "params": [
            "breakout_len", "momentum_len", "momentum_thresh", "trend_len",
            "atr_len", "min_atr", "trailing_stop_pct", "max_hold_bars"
        ],
        "universal": ["initial_capital", "qty"]
    },
    "SMA Cross": {
        "function": sma_cross_strategy,
        "params": [
            "fast_len", "slow_len"
        ],
        "universal": ["initial_capital", "qty"]
    }
}


# ===========================================
# 4. DASH APP LAYOUT
# ===========================================
# Build the web UI. The left column is for inputs (data, strategy, parameters),
# the right column for results (tables, plots).app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Universal Backtester (Dash)")),
                dbc.CardBody([
                    html.Div(id="upload-btn-container"),
                    html.Br(),
                    dcc.Dropdown(
                        id='feed-choice',
                        options=[{"label": f, "value": f} for f in ["alpaca", "csv", "tvdatafeed", "ccxt", "yfinance", "tradingview_ta"]],
                        placeholder="Select data source...",
                        clearable=False,
                    ),
                    dcc.Input(id="symbol-input", type="text", placeholder="Symbol", value="RGTI"),
                    dcc.Dropdown(id='tf-dropdown', options=[{"label": tf, "value": tf} for tf in ["1m", "3m", "5m", "15m", "30m", "1h", "1d"]], value="3m"),
                    html.Br(),
                    html.Button("Load Data", id="btn-load-data", className="mb-2"),
                    html.Div(id="data-status"),
                    html.Hr(),
                    html.H5("Session & Universal Params", className="mt-2"),
                    dbc.Row([
                        dbc.Col(dcc.Input(id="session-start", type="text", value="13:31", placeholder="Session Start")),
                        dbc.Col(dcc.Input(id="session-end", type="text", value="19:52", placeholder="Session End")),
                    ]),
                    dcc.Input(id="initial-capital", type="number", value=200000, placeholder="Initial Capital"),
                    dcc.Input(id="qty", type="number", value=1, placeholder="Qty"),
                    dcc.Input(id="maxtradesperday", type="number", value=1, placeholder="Max Trades/Day"),
                    dcc.Checklist(
                        id="use-session-end-rule", options=[{"label": "Session End Exit Rule", "value": "yes"}], value=["yes"]),
                    html.Hr(),
                    html.H5("Strategy"),
                    dcc.Dropdown(id="strategy-choice", options=[{"label": k, "value": k} for k in STRATEGY_REGISTRY.keys()],
                                 value=list(STRATEGY_REGISTRY.keys())[0]),
                    html.Div(id="strategy-params-ui"),  # parameter widgets
                    html.Button("Run Optimization", id="btn-optimize", className="mt-2"),
                    html.Div(id="opt-status"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(html.Button("💾 Save Best Parameters", id="btn-save-params", className="w-100 mb-1"), width=6),
                        dbc.Col(html.Button("📥 Load Best Parameters", id="btn-load-params", className="w-100 mb-1"), width=6),
                    ]),
                    html.Div(id="load-table-container"),
                    dcc.Store(id="loaded-data-store"),
                    dcc.Store(id="best-params-store"),
                    dcc.Store(id="trades-store"),
                ]),
            ], className="shadow mb-3", style={"background": "#1a233a", "border-radius": "20px"}),
        width=3),

        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("Results")),
                dbc.CardBody([
                    html.Div(id="results-summary"),
                    html.Hr(),
                    html.H4("Trade Log"),
                    dash_table.DataTable(id="trade-log-table", page_size=15, style_cell={"backgroundColor": "#101728", "color": "#F6F8FA"}),
                    # html.H4("Equity/Price Chart"),  # Optionally hide chart if not needed
                    # dcc.Graph(id="price-chart"),
                ])
            ], className="shadow mb-3", style={"background": "#232946", "border-radius": "20px"}),
        width=9)
    ])
], fluid=True)

@app.callback(
    Output("upload-btn-container", "children"),
    Input("feed-choice", "value"),
)
def show_upload_btn(feed_choice):
    print("Dropdown value in upload-btn callback:", feed_choice)
    if feed_choice == "csv":
        return dcc.Upload(id='upload-data', children=html.Button('Upload CSV'), multiple=False)
    return ""

# ===========================================
# 5. CALLBACK: Data Loading
# ===========================================
# Loads CSV or API data when user clicks Load Data. Data is stored in dcc.Store.

@app.callback(
    Output('data-status', 'children'),
    Output('loaded-data-store', 'data'),
    Input('btn-load-data', 'n_clicks'),
    State('feed-choice', 'value'),
    State('symbol-input', 'value'),
    State('tf-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(n, feed, symbol, tf, upload_contents, upload_filename):
    if not n:
        return "", None
    df = None
    msg = ""
    try:
        if feed == "csv" and upload_contents:
            import base64, io
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.BytesIO(decoded))
            msg = f"CSV loaded: {upload_filename}"
        elif feed == "yfinance":
            df = load_yfinance(symbol, tf, "2025-07-01", "2025-12-31")
            msg = f"Loaded from yfinance: {symbol} {tf}"
        elif feed == "alpaca":
            # You should provide keys securely for production!
            key, secret, url = "demo", "demo", "https://paper-api.alpaca.markets"
            df = load_alpaca(symbol, tf, "2025-07-01", "2025-12-31", key, secret, url)
            msg = f"Loaded from Alpaca: {symbol} {tf}"
        elif feed == "tvdatafeed":
            df = load_tvdatafeed(symbol, "NASDAQ", tf, 500)
            msg = f"Loaded from TVDatafeed: {symbol} {tf}"
        elif feed == "ccxt":
            df = load_ccxt(symbol=symbol, timeframe=tf, limit=500)
            msg = f"Loaded from CCXT: {symbol} {tf}"
        elif feed == "tradingview_ta":
            df = get_tv_ta(symbol, exchange="NASDAQ", interval=tf)
            msg = f"Loaded from TradingView TA: {symbol} {tf}"
    except Exception as e:
        msg = f"Error loading: {e}"
    if df is not None:
        df = df[[c for c in df.columns if c.lower() not in ("", "index")]]
        return msg, df.to_json(date_format="iso", orient="split")
    return msg, None


# ===========================================
# 6. CALLBACK: Strategy Parameter UI Generation
# ===========================================
# Dynamically builds parameter widgets for the chosen strategy, using best or default values.

@app.callback(
    Output('strategy-params-ui', 'children'),
    Input('strategy-choice', 'value'),
    State('best-params-store', 'data')
)
def show_strategy_params(strategy, best_params):
    # Build a widget for each strategy parameter (use last used/best or a default)
    params = STRATEGY_REGISTRY[strategy]["params"]
    fields = []
    for p in params:
        default = 10
        if best_params and p in best_params:
            default = best_params[p]
        fields.append(
            dbc.Row([
                dbc.Col(html.Label(p), width=6),
                dbc.Col(dcc.Input(id=f"param-{p}", type="number", value=default), width=6),
            ])
        )
    return fields

# ===========================================
# 7. CALLBACK: Run Optimization / Backtest
# ===========================================
# Runs the selected strategy/backtest when user clicks Run Optimization.
# Stores the best parameters and trade log in dcc.Store for later use.

@app.callback(
    Output('opt-status', 'children'),
    Output('best-params-store', 'data'),
    Output('trades-store', 'data'),
    Input('btn-optimize', 'n_clicks'),
    State('loaded-data-store', 'data'),
    State('strategy-choice', 'value'),
    State('session-start', 'value'),
    State('session-end', 'value'),
    State('initial-capital', 'value'),
    State('qty', 'value'),
    State('maxtradesperday', 'value'),
    State('use-session-end-rule', 'value'),
    State('strategy-params-ui', 'children')
)
def run_optimizer(n, data_json, strategy, session_start, session_end, initial_capital, qty, maxtrades, use_end_rule, param_children):
    if not n or not data_json:
        return "", None, None
    df = pd.read_json(data_json, orient="split")
    # Assemble universal and param kwargs
    universal_kwargs = dict(
        session_start=session_start,
        session_end=session_end,
        keeplime=True,
        initial_capital=initial_capital,
        qty=qty,
        maxtradesperday=maxtrades,
        use_session_end_rule="yes" in (use_end_rule or [])
    )
    params = STRATEGY_REGISTRY[strategy]["params"]
    param_vals = {}
    for c in param_children:
        pid = c['props']['children'][1]['props']['id']
        val = c['props']['children'][1]['props']['value']
        param_name = pid.replace("param-", "")
        param_vals[param_name] = val
    call_kwargs = {**universal_kwargs, **param_vals}
    # Call strategy
    try:
        trades, df_all = STRATEGY_REGISTRY[strategy]["function"](df, **call_kwargs)
        best_params = param_vals
        return f"Optimization run complete.", best_params, trades.to_json(date_format="iso", orient="split")
    except Exception as e:
        return f"Error during optimization: {e}", None, None

# ===========================================
# 8. CALLBACK: Display Results, Trade Log, Chart
# ===========================================
# Shows summary stats, trades table, and price/equity chart after optimization.

@app.callback(
    Output('results-summary', 'children'),
    Output('trade-log-table', 'data'),
    Output('trade-log-table', 'columns'),
    Output('price-chart', 'figure'),
    Input('trades-store', 'data')
)
def display_results(trades_json):
    if not trades_json:
        return "", [], [], go.Figure()
    trades = pd.read_json(trades_json, orient="split")
    summary = [
        html.P(f"Total Trades: {len(trades)}"),
        html.P(f"Total Return: {trades['return'].sum():.2f}%") if "return" in trades else "",
        html.P(f"Win Rate: {((trades['pnl'] > 0).mean() * 100):.2f}%" if "pnl" in trades else ""),
    ]
    columns = [{"name": c, "id": c} for c in trades.columns]
    fig = go.Figure()
    if "EntryTime" in trades.columns and "EntryPrice" in trades.columns:
        fig.add_trace(go.Scatter(x=trades["EntryTime"], y=trades["EntryPrice"], mode='lines+markers', name="Entries"))
    return summary, trades.to_dict("records"), columns, fig


# ===========================================
# 9. CALLBACK: Save Best Parameters
# ===========================================
# Stores best parameters and results for current strategy/symbol.

@app.callback(
    Output('best-params-store', 'data'),
    Output('load-table-container', 'children'),
    Input('btn-save-params', 'n_clicks'),
    Input('btn-load-params', 'n_clicks'),
    State('strategy-choice', 'value'),
    State('symbol-input', 'value'),
    State('best-params-store', 'data'),
    State('trades-store', 'data')
)
def save_or_load_params(n_save, n_load, strategy, symbol, params, trades_json):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, ""
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn_id == 'btn-save-params':
        if not n_save or not params:
            return dash.no_update, ""
        # Main stats
        trades = pd.read_json(trades_json, orient="split") if trades_json else pd.DataFrame()
        results = {
            "Total Return": float(trades['return'].sum()) if not trades.empty and 'return' in trades else 0,
            "Win Rate": float((trades['pnl'] > 0).mean()) if not trades.empty and 'pnl' in trades else 0,
            "Total Trades": int(len(trades))
        }
        register_params(strategy, symbol, params, results)
        return dash.no_update, html.Div("Parameters saved!", style={"color": "green"})
    elif btn_id == 'btn-load-params':
        if not n_load:
            return dash.no_update, ""
        saved = get_saved_params(strategy, symbol)
        if saved:
            params = saved["params"]
            df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
            result_df = None
            if "results" in saved:
                result_df = pd.DataFrame(list(saved["results"].items()), columns=["Metric", "Value"])
            tables = [
                html.H5("Loaded Parameters"),
                dash_table.DataTable(df.to_dict('records'), columns=[{"name": i, "id": i} for i in df.columns]),
            ]
            if result_df is not None:
                tables.append(html.H5("Saved Results"))
                tables.append(dash_table.DataTable(result_df.to_dict('records'), columns=[{"name": i, "id": i} for i in result_df.columns]))
            return params, tables
        else:
            return dash.no_update, html.Div("No saved parameters found.", style={"color": "red"})
    return dash.no_update, ""

# ===========================================
# 11. MAIN ENTRY POINT
# ===========================================

if __name__ == '__main__':
    app.run(debug=True)
