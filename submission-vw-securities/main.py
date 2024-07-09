# Trend following using EMA

import numpy as np

def EMA(prices, window):
    alpha = 2 / (window + 1)
    emas = np.zeros_like(prices)
    emas[:, :window] = prices[:, :window]  # Initialize first 'window' days

    for t in range(window, prices.shape[1]):
        emas[:, t] = alpha * prices[:, t] + (1 - alpha) * emas[:, t-1]

    return emas

def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)

    if nt >= 30:  # Make sure there is enough data
        ema_short = EMA(prices, 5)[:, -1]  # Last value of the short-term EMA
        ema_long = EMA(prices, 97)[:, -1]  # Last value of the long-term EMA

        # Generate buy or sell signals
        buy_signals = ema_short > ema_long
        sell_signals = ema_short < ema_long
        signals = buy_signals.astype(int) - sell_signals.astype(int)

        # Calculate the maximum position size based on the latest price
        max_pos_size = (10000 / prices[:, -1]).astype(int)

        # Apply signals and adjust for position limits
        positions = signals * max_pos_size
        positions = np.clip(positions, -max_pos_size, max_pos_size)

    return positions

