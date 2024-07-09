import numpy as np
import pandas as pd
import statsmodels.api as sm
from collections import defaultdict
from statsmodels.tsa.stattools import coint

nInst = 50
currentPos = np.zeros(nInst)

# Cache
pairs = defaultdict(None)

# Config
window1 = 5
window2 = 60
max_window = max(window1, window2)

max_cash_limit = 500

z_action_threshold = 2
z_neutral_threshold = 0.2

def find_cointegrated_pairs(data: pd.DataFrame):
    nInst = data.shape[1]
    score_matrix = np.zeros((nInst, nInst))
    pvalue_matrix = np.ones((nInst, nInst))
    keys = data.keys()
    pairs = []
    for i in range(nInst):
        for j in range(i+1, nInst):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return pairs

def trade(S1: pd.Series, S2: pd.Series, window1=window1, window2=window2):
    if window1 == 0 or window2 == 0:
        return 0, 0  # Return neutral positions if windows are incorrectly set.

    # ratios = S1 / S2
    # ma1 = ratios.rolling(window=window1, center=False).mean()
    # ma2 = ratios.rolling(window=window2, center=False).mean()
    # std = ratios.rolling(window=window2, center=False).std()
    # zscore = (ma1 - ma2) / std
    # zscore = zscore.iloc[-1]

    diffs = np.abs(S1 - S2)
    mean = diffs.mean()
    std = diffs.std()
    curr = diffs.iloc[-1]
    zscore = np.abs(curr - mean) / std

    # Latest price
    latest_price_s1 = S1.iloc[-1]
    latest_price_s2 = S2.iloc[-1]

    # Calculate maximum allowable positions based on $10,000 limit and current prices
    max_pos_S1 = max_cash_limit / latest_price_s1
    max_pos_S2 = max_cash_limit / latest_price_s2

    # Initialize positions
    countS1 = currentPos[S1.name]
    countS2 = currentPos[S2.name]

    # Determine positions based on z-score thresholds
    if zscore > z_action_threshold and latest_price_s1 > latest_price_s2:          # Ratio is too high - short S1, long S2
        countS1 = -int(max_pos_S1)
        countS2 = int(max_pos_S2)
    elif zscore > z_action_threshold and latest_price_s2 > latest_price_s1:       # Ratio is too low - long S1, short S2
        countS2 = -int(max_pos_S2)
        countS1 = int(max_pos_S1)
    elif abs(zscore) < z_neutral_threshold:  # Neutral zone, close positions
        countS1 = 0
        countS2 = 0

    return countS1, countS2, zscore

def getMyPosition(prcSoFar):
    global pairs

    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    if not pairs:
        pairs = find_cointegrated_pairs(data)

    for (stock_1, stock_2) in pairs:
    # for (stock_1, stock_2) in [pairs[0]]:
        # print(stock_1, stock_2)
        idx1, idx2 = data.columns.get_loc(stock_1), data.columns.get_loc(stock_2)
        S1, S2 = data.iloc[-max_window:, idx1], data.iloc[-max_window:, idx2]
        currentPos[idx1], currentPos[idx2], zscore = trade(S1, S2)

    return currentPos, zscore
