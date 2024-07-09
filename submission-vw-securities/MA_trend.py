# Trend Following Strategy using MA

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    # Ensure there's enough data to compute both SMAs
    if nt < 2:
        return np.zeros(nInst)

    # Parameters for SMA
    short_window = 5  # Short-term moving average (e.g., 20 days)
    long_window = 97   # Long-term moving average (e.g., 50 days)

    # Check if there are enough days to calculate both moving averages
    if nt < long_window:
        # Not enough data, return current position
        return currentPos

    # Calculate short and long SMAs
    sma_short = np.mean(prcSoFar[:, -short_window:], axis=1)
    sma_long = np.mean(prcSoFar[:, -long_window:], axis=1)

    # Calculate maximum number of shares for each stock based on the limit of $10,000 exposure per stock
    max_position_per_stock = np.floor(10000 / prcSoFar[:, -1])

    # Determine positions based on the crossover strategy
    # Buy when the short SMA crosses above the long SMA
    # Sell when the short SMA crosses below the long SMA
    new_positions = np.where(sma_short > sma_long, max_position_per_stock, 0)
    new_positions = np.where(sma_short < sma_long, -max_position_per_stock, new_positions)

    # Update current positions
    currentPos = new_positions

    return currentPos.astype(int)
