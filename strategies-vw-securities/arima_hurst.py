import numpy as np
import pandas as pd
from collections import defaultdict

from hurst import compute_Hc
from pmdarima import auto_arima
from pmdarima.arima import ndiffs

# Setup
nInst = 50
currentPos = np.zeros(nInst)
lastPrice = np.zeros(nInst)
model_cache = defaultdict(None)
hurst_cache = defaultdict(int)

# Config
take_profit = 0.10    # minimum $0.10 profit
stop_loss = 0.

forecast_window = 10     # forecast 10 days into the future
hurst_threshold = 0.3

#################################

def hurst(prices, window):
    # Evaluate Hurst equation
    assert np.abs(window) >= 100
    (H, c, data) = compute_Hc(prices[window:], kind='price', simplified=True)
    return H

def is_mean_reverting(H, threshold):
    """Returns True if anti-persistent behaviour detected"""
    return H < threshold

def is_persistent(H, threshold):
    """Returns True if persistent behaviour detected"""
    return H > (1 - threshold)

def not_brownian(H, threshold):
    """Returns True if behaviour detected is not Brownian motion"""
    return is_mean_reverting(H, threshold) or is_persistent(H, threshold)

def get_regime(H, threshold):
    if is_mean_reverting(H, threshold):
        return 1
    if is_persistent(H, threshold):
        return 2
    return 3

#################################

def arima_train(train_prices: pd.Series, stock_name: str) -> None:
    kpss_diffs = ndiffs(train_prices, alpha=0.05, test='kpss', max_d=20)
    adf_diffs = ndiffs(train_prices, alpha=0.05, test='adf', max_d=20)
    n_diffs = max(adf_diffs, kpss_diffs)

    model = auto_arima(train_prices, d=n_diffs, seasonal=True, stepwise=True,
                        suppress_warnings=True, error_action='ignore',
                        max_p=1, start_q=20, max_q=80,
                        max_order=None, trace=False)
    model_cache[stock_name] = model

def arima_update(observations: pd.Series, stock_name: str) -> None:
    model_cache[stock_name].update(observations)

def arima_forecast(stock_name, window=10) -> None | tuple[np.array, np.array]:
    return model_cache[stock_name].predict(n_periods=window).tolist()[-1]

#################################

def position_value(current_position, current_price, buy_price):
    if current_position > 0:
        return current_price - buy_price
    if current_position < 0:
        return buy_price - current_price
    return 0

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
        # Also retrain ARIMA model if Hurst regime changed DEPRECATED
        if stock_name not in model_cache:
            arima_train(prices, stock_name)
        else:
            arima_update(prices.iloc[-forecast_window:], stock_name)

        # If take profit or stop loss, close position for today
        position_val = position_value(currentPos[i], prices.iloc[-1], lastPrice[i])
        if (position_val > take_profit or position_val < stop_loss) and False:
            currentPos[i] = 0
        else:
            # Inform strategy using Hurst
            H = hurst(prices, -100)

            # Get the latest price, and predict window periods into the future
            lastPrice[i] = prices.iloc[-1]
            price_forecast = arima_forecast(stock_name, window=forecast_window)

            # Only trade if predicted profit surpasses threshold
            predicted_profit = price_forecast - lastPrice[i]
            if np.abs(predicted_profit) >= take_profit and not_brownian(H, hurst_threshold):
                currentPos[i] = np.sign(predicted_profit) * 300

    return currentPos
