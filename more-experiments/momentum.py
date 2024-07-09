# Momentum trading
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    # Ensure there's enough data to compute momentum
    if nt < 2:
        return np.zeros(nInst)

    # Parameters
    lookback = 2  # Look back 5 days to calculate momentum
    upper_threshold = 0.06  # 4% increase for buying
    lower_threshold = -0.02  # 2% decrease for selling

    # Calculate momentum based on closing prices lookback days ago
    if nt < lookback + 1:
        # Not enough data to calculate momentum, return current position
        return currentPos

    # Compute percentage change for momentum
    momentum = (prcSoFar[:, -1] - prcSoFar[:, -lookback]) / prcSoFar[:, -lookback]

    # Calculate maximum number of shares for each stock based on current price
    # and the limit of $10,000 exposure per stock
    max_position_per_stock = np.floor(10000 / prcSoFar[:, -1])

    # Determine positions based on momentum and thresholds
    new_positions = np.zeros(nInst)
    new_positions = np.where(momentum > upper_threshold, max_position_per_stock, new_positions)
    new_positions = np.where(momentum < lower_threshold, -max_position_per_stock, new_positions)

    # Update current positions
    currentPos = new_positions

    return currentPos.astype(int)
