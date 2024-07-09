import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

nInst = 50
currentPos = np.zeros(nInst)

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

def trade(S1, S2, window1=5, window2=60):
    if window1 == 0 or window2 == 0:
        return 0, 0  # Return neutral positions if windows are incorrectly set.

    # Convert arrays to pandas Series if they aren't already
    if isinstance(S1, np.ndarray):
        S1 = pd.Series(S1)
    if isinstance(S2, np.ndarray):
        S2 = pd.Series(S2)
    
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1, center=False).mean()
    ma2 = ratios.rolling(window=window2, center=False).mean()
    std = ratios.rolling(window=window2, center=False).std()
    zscore = (ma1 - ma2) / std

    # Initialize positions
    countS1 = 0
    countS2 = 0

    # Determine positions based on z-score thresholds
    for i in range(len(zscore)):
        # Calculate maximum allowable positions based on $10,000 limit and current prices
        max_pos_S1 = 10000 / S1[i]
        max_pos_S2 = 10000 / S2[i]

        if zscore[i] > 1:  # Buy signal for S1
            countS1 = int(max_pos_S1)  # Buy up to the maximum allowable position
            countS2 = -int(max_pos_S2)  # Corresponding sell for S2
        elif zscore[i] < -1:  # Sell signal for S2
            countS1 = -int(max_pos_S1)  # Sell up to the maximum allowable position
            countS2 = int(max_pos_S2)  # Corresponding buy for S1
        elif abs(zscore[i]) < 0.75:  # Neutral zone, close positions
            countS1 = 0
            countS2 = 0

    return countS1, countS2


def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape
    data = pd.DataFrame(prcSoFar.T)
    
    _, _, pairs = find_cointegrated_pairs(data)
    
    for pair in pairs:
        idx1, idx2 = data.columns.get_loc(pair[0]), data.columns.get_loc(pair[1])
        S1, S2 = data.iloc[:, idx1], data.iloc[:, idx2]
        currentPos[idx1], currentPos[idx2] = trade(S1, S2)

    return currentPos
