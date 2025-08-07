from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

api_key = "PK99OKUJAAI9KIINMT9G"
api_secret = "isFedDOxxFjl5KWuzSnPoe8Zb7aHMUYig4VDsuyI"
api_url = "https://paper-api.alpaca.markets"

client = StockHistoricalDataClient(api_key, api_secret, api_url)
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,
    start=datetime(2024, 6, 1),
    end=datetime(2024, 6, 5)
)
bars = client.get_stock_bars(request_params)
print(bars.df)
