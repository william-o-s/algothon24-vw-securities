# Mean reversion to tune the hyper-parameter

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar, n1=30, base_offset=0.01):
    global currentPos
    nInst, nt = prcSoFar.shape
    
    if nt < 2:
        return np.zeros(nInst)
    
    # Calculate moving average and standard deviation for volatility adjustment
    ma = np.mean(prcSoFar[:, -n1:], axis=1)
    std_dev = np.std(prcSoFar[:, -n1:], axis=1)

    # Adjust offset based on recent volatility
    offset = base_offset + std_dev / ma

    # Calculate positions
    max_position = 5000 / prcSoFar[:, -1]
    new_positions = np.zeros(nInst)
    for i in range(nInst):
        price = prcSoFar[i, -1]
        if price < ma[i] * (1 - offset[i]):
            new_positions[i] = max_position[i]
        elif price > ma[i] * (1 + offset[i]):
            new_positions[i] = -max_position[i]
    
    currentPos += new_positions
    currentPos = np.clip(currentPos, -max_position, max_position)
    
    return currentPos.astype(int)
