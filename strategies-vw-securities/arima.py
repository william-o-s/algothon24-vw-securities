import numpy as np
import pandas as pd
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from collections import defaultdict

# Setup
nInst = 50
currentPos = np.zeros(nInst)
cache = defaultdict(None)

# Config
threshold = 0.10   # minimum $0.10 profit

def apply_arima(prices, stock_name):
    """
    Apply ARIMA model to predict stock prices.

    Parameters:
    - prices: Series, input stock data

    Returns:
    - prediction: int, ARIMA-predicted stock price
    """
    if stock_name in cache:
        model = cache[stock_name]
        model.update(prices.iloc[-1])
    else:
        kpss_diffs = ndiffs(prices, alpha=0.05, test='kpss', max_d=20)
        adf_diffs = ndiffs(prices, alpha=0.05, test='adf', max_d=20)
        n_diffs = max(adf_diffs, kpss_diffs)

        model = auto_arima(prices, d=n_diffs, seasonal=True, stepwise=True,
                            suppress_warnings=True, error_action='ignore',
                            max_p=10, max_q=10,
                            max_order=None, trace=True)
        cache[stock_name] = model

    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)

    return fc.tolist()[0], np.asarray(conf_int).tolist()[0]

def getMyPosition(prcSoFar):
    global currentPos

    prcSoFar = pd.DataFrame(prcSoFar.T)
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)

    # Apply ARIMA predictions
    for i, (stock_name, prices) in enumerate(prcSoFar.items()):
        price_prediction, _ = apply_arima(prices, stock_name)
        last_price = prices.iloc[-1]

        predicted_profit = price_prediction - last_price

        if np.abs(predicted_profit) >= threshold:
            currentPos[i] = np.sign(predicted_profit)

    return currentPos
