import numpy as np
import pandas as pd
from collections import defaultdict
from math import comb, perm
from itertools import combinations, permutations
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, add_constant, adfuller

### Algothon ###
nInst = 50
commRate = 0.0010
dlrPosLimit = 10000
currentPos = np.zeros(nInst)
################

### Pairs Trading ###
rolling_window = 50                     # Toggle

pairs_cache = None
beta_matrix = None
#####################

window1 = 5
window2 = 20

zscore_action_threshold = 1             # Toggle
zscore_neutral_threshold = 0.25         # Toggle

#######################################################################################################################

def coint_simple(S1, S2):
    result = coint(S1, S2)
    score, pvalue = result[0], result[1]
    return pvalue, score

def coint_spread(S1, S2):
    X = add_constant(S2)
    results = OLS(S1, X).fit()
    beta = results.params[S2.name]
    # constant = results.params['const']

    residuals = S1 - beta * S2
    pvalue = adfuller(residuals)[1]
    return pvalue, beta

def find_cointegrated_pairs(df: pd.DataFrame, significance=0.005):
    nInst = df.shape[1]
    score_matrix = np.zeros((nInst, nInst))
    beta_matrix = np.ones((nInst, nInst))
    pvalue_matrix = np.ones((nInst, nInst))

    # possible_pairs = combinations(df.columns, 2)
    possible_pairs = permutations(df.columns, 2)    # Pairs can be order-dependent based on test

    pairs = []
    for S1, S2 in possible_pairs:
        simple_pvalue, score = coint_simple(df[S1], df[S2])
        score_matrix[S1, S2] = score

        spread_pvalue, beta = coint_spread(df[S1], df[S2])
        beta_matrix[S1, S2] = beta

        if simple_pvalue < significance and spread_pvalue < significance:
            pvalue_matrix[S1, S2] = max(simple_pvalue, spread_pvalue)   # take the more unconfident pvalue
            pairs.append((S1, S2))

    return pairs, score_matrix, beta_matrix

#######################################################################################################################

def trade(S1: pd.Series, S2: pd.Series):
    stock1, stock2 = S1.name, S2.name

    # Calculate hedge ratio
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1, center=False).mean()
    ma2 = ratios.rolling(window=window2, center=False).mean()
    std = ratios.rolling(window=window2, center=False).std()
    zscore = (ma1 - ma2) / std
    zscore = zscore.iloc[-1]

    # Latest price
    latest_price_s1 = S1.iloc[-1]
    latest_price_s2 = S2.iloc[-1]

    # Calculate maximum allowable positions based on $10,000 limit or Kelly criterion
    rules_pos_S1 = 10_000 / latest_price_s1
    rules_pos_S2 = 10_000 / latest_price_s2

    max_pos_S1 = rules_pos_S1 * 0.5
    max_pos_S2 = rules_pos_S2 * 0.5

    # Initialize positions
    countS1 = currentPos[stock1]
    countS2 = currentPos[stock2]

    if currentPos[stock1] != 0 and currentPos[stock2] != 0:
        return countS1, countS2, zscore

    if zscore >= zscore_action_threshold:
        countS1 = -int(max_pos_S1)
        countS2 = int(max_pos_S2)
    elif zscore <= -zscore_action_threshold:
        countS1 = int(max_pos_S1)
        countS2 = -int(max_pos_S2)
    elif np.abs(zscore) <= zscore_neutral_threshold:
        countS1 = countS2 = 0

    return countS1, countS2, zscore

#######################################################################################################################

def getMyPosition(prcSoFar):
    global pairs_cache, beta_matrix

    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    if not pairs_cache:
        pairs_cache, _, beta_matrix = find_cointegrated_pairs(data)
        print(len(pairs_cache))

    # for (stock_1, stock_2) in pairs_cache:
    for (stock_1, stock_2) in [pairs_cache[0]]:
    # for (stock_1, stock_2) in [(0, 1)]:
        # print(stock_1, stock_2)
        S1, S2 = data[stock_1].iloc[-rolling_window:], data[stock_2].iloc[-rolling_window:]
        currentPos[stock_1], currentPos[stock_2], zscore = trade(S1, S2)

    return currentPos, zscore
