import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from collections import defaultdict

# Setup
nInst = 50
currentPos = np.zeros(nInst)
lastPrice = np.zeros(nInst)
model_cache = defaultdict(None)

# Config
threshold = 0.10    # minimum $0.10 profit
forecast_window = 20     # forecast 20 days into the future
stop_loss = 0.

def arima_train(train_prices: pd.Series, stock_name: str) -> None:
    kpss_diffs = ndiffs(train_prices, alpha=0.05, test='kpss', max_d=20)
    adf_diffs = ndiffs(train_prices, alpha=0.05, test='adf', max_d=20)
    n_diffs = max(adf_diffs, kpss_diffs)

    model = auto_arima(train_prices, d=n_diffs, seasonal=True, stepwise=True,
                        suppress_warnings=True, error_action='ignore',
                        max_p=10, max_q=10,
                        max_order=None, trace=True)
    model_cache[stock_name] = model

def arima_update(observations: pd.Series, stock_name: str) -> None:
    model_cache[stock_name].update(observations)

def arima_forecast(stock_name, window=10) -> None | tuple[np.array, np.array]:
    return model_cache[stock_name].predict(n_periods=window).tolist()[-1]

def getMyPosition(prcSoFar):
    prcSoFar = pd.DataFrame(prcSoFar.T)
    (nt, _) = prcSoFar.shape
    if (nt < 2):
        return currentPos

    if nt % forecast_window != 0:
        return currentPos

    # Apply ARIMA predictions
    for i, (stock_name, prices) in enumerate(prcSoFar.items()):
        # Create ARIMA model if not exists, otherwise update with last window observations
        if stock_name not in model_cache:
            arima_train(prices, stock_name)
        else:
            arima_update(prices.iloc[-forecast_window:], stock_name)

        # Get the latest price, and predict window periods into the future
        lastPrice[i] = prices.iloc[-1]
        price_forecast = arima_forecast(stock_name, window=forecast_window)

        # Only trade if predicted profit surpasses threshold
        predicted_profit = price_forecast - lastPrice[i]
        if np.abs(predicted_profit) >= threshold:
            currentPos[i] = np.sign(predicted_profit) * 300

    return currentPos
