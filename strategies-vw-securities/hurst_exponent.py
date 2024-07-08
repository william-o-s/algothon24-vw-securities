import numpy as np
import pandas as pd
from hurst import compute_Hc

# Setup
nInst = 50
currentPos = np.zeros(nInst)
boughtPrice = np.zeros(nInst)

def hurst(prices, window):
    # Evaluate Hurst equation
    (H, c, data) = compute_Hc(prices[window:], kind='price', simplified=True)
    return H

def should_trade(H, threshold):
    """Returns True if persistent behaviour detected"""
    return H > (1 - threshold)

def getMyPosition(prcSoFar, window=-200, threshold=0.45, take_profit=0.10, stop_loss=0.05):
    global currentPos
    global boughtPrice

    prcSoFar = pd.DataFrame(prcSoFar.T)
    (nt, nins) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)

    # Apply Hurst predictions
    for i, (_, prices) in enumerate(prcSoFar.items()):
        H = hurst(prices, window)
        latest_price = prices.iloc[-1]

        if should_trade(H, threshold):
            earliest_price = prices.iloc[window]
            currentPos[i] = np.sign(latest_price - earliest_price) * 300
            boughtPrice[i] = latest_price
        else:
            profit = (latest_price / boughtPrice[i]) - 1
            if profit >= take_profit or -profit >= stop_loss:
                currentPos[i] = 0

    return currentPos
