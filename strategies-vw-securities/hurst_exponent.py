import numpy as np
import pandas as pd
from hurst import compute_Hc

# Setup
nInst = 50
commRate = 0.0010
currentPos = np.zeros(nInst)
boughtPrice = np.zeros(nInst)

def hurst(prices, window):
    # Evaluate Hurst equation
    (H, c, data) = compute_Hc(prices[window:], kind='price', simplified=True)
    return H

def is_mean_reverting(H, threshold):
    """Returns True if anti-persistent behaviour detected"""
    return H < threshold

def is_persistent(H, threshold):
    """Returns True if persistent behaviour detected"""
    return H > (1 - threshold)

def kelly(H, threshold, margin):
    p = np.abs(H - threshold) / margin
    b = 1 / commRate
    return p - ((1 - p) / b)

def getMyPosition(prcSoFar, window=-125, threshold=0.3, take_profit=0.10, stop_loss=0.04, base=700, interval=10):
    global currentPos
    global boughtPrice

    prcSoFar = pd.DataFrame(prcSoFar.T)
    (nt, nins) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    elif nt % interval != 0:
        return currentPos   # only make moves every 10 days

    # Apply Hurst predictions
    for i, (_, prices) in enumerate(prcSoFar.items()):
        H = hurst(prices, window)
        latest_price = prices.iloc[-1]
        window_prices = prices.iloc[window:]

        if is_mean_reverting(H, threshold):
            # H < threshold
            currentPos[i] = np.sign(window_prices.mean() - latest_price) * base * kelly(H, threshold, threshold)
            boughtPrice[i] = latest_price
        elif is_persistent(H, threshold):
            # H > (1 - threshold)
            earliest_price = prices.iloc[window]
            currentPos[i] = np.sign(latest_price - earliest_price) * base * kelly(H, 1 - threshold, threshold)
            boughtPrice[i] = latest_price
        else:
            # threshold < H < (1 - threshold)
            profit = (latest_price / boughtPrice[i]) - 1 if boughtPrice[i] else 0
            if profit >= take_profit or -profit >= stop_loss:
                currentPos[i] = 0

    return currentPos
