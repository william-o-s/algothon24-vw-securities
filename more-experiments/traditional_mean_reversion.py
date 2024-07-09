# Mean reversion from the Internet

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape
    
    if nt < 2:
        return np.zeros(nInst)
    
    # Parameters
    n1 = 30  # Period for the moving average
    offset = 0.015  # Buy/sell offset from the moving average
    
    # Calculate rolling mean avoiding future data leakage
    if nt < n1:
        # Not enough data to compute n1 day MA
        ma = np.nanmean(prcSoFar, axis=1)
    else:
        ma = np.mean(prcSoFar[:, -n1:], axis=1)
    
    # Positions calculation
    # Calculating the position size based on $5000 divided by last price to stay within position limits
    max_position = 5000 / prcSoFar[:, -1]
    new_positions = np.zeros(nInst)
    
    # Generate positions based on the mean reversion logic
    for i in range(nInst):
        price = prcSoFar[i, -1]
        moving_avg = ma[i]
        
        if price < moving_avg * (1 - offset):
            # Price is more than offset% below MA, consider buying
            new_positions[i] = max_position[i]
        elif price > moving_avg * (1 + offset):
            # Price is more than offset% above MA, consider selling
            new_positions[i] = -max_position[i]
    
    # Update current positions
    currentPos += new_positions
    
    # Clipping positions to respect the $10k limit (handled externally as well)
    currentPos = np.clip(currentPos, -max_position, max_position)
    
    return currentPos.astype(int)
