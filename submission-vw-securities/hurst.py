# This one does not work and does not trade
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def EMA(prices, window):
    """ Calculate Exponential Moving Average for given prices and window. """
    alpha = 2 / (window + 1)
    emas = np.zeros_like(prices)
    emas[:, :window] = prices[:, :window]  # Initialize first 'window' days
    for t in range(window, prices.shape[1]):
        emas[:, t] = alpha * prices[:, t] + (1 - alpha) * emas[:, t-1]
    return emas

def hurst_exponent(ts, max_lags=100):
    """ Calculate the Hurst Exponent of a time series. """
    lags = range(2, max_lags)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape
    
    if nt < 100:  # Ensure there's enough data to compute Hurst Exponent
        return np.zeros(nInst)
    
    # Parameters for trading
    hurst_threshold = 0.5  # Threshold to distinguish between trending and mean-reverting
    short_window = 20  # EMA short period
    long_window = 50  # EMA long period
    sma_window = 30  # SMA period for mean reversion
    offset = 0.015  # Offset for mean reversion

    # Compute the Hurst Exponent for each instrument
    hurst_exponents = np.array([hurst_exponent(prcSoFar[i, -100:]) for i in range(nInst)])

    # Decide the trading strategy based on Hurst Exponent
    for i in range(nInst):
        if hurst_exponents[i] > hurst_threshold:
            # Trend following using EMA
            ema_short = EMA(prcSoFar[i:i+1], short_window)[-1, -1]
            ema_long = EMA(prcSoFar[i:i+1], long_window)[-1, -1]
            if ema_short > ema_long:
                currentPos[i] = 10000 / prcSoFar[i, -1]
            elif ema_short < ema_long:
                currentPos[i] = -10000 / prcSoFar[i, -1]
        elif hurst_exponents[i] < hurst_threshold:
            # Mean reversion using SMA
            ma = np.mean(prcSoFar[i, -sma_window:], axis=0)
            price = prcSoFar[i, -1]
            if price < ma * (1 - offset):
                currentPos[i] = 10000 / price
            elif price > ma * (1 + offset):
                currentPos[i] = -10000 / price

    return currentPos.astype(int)
