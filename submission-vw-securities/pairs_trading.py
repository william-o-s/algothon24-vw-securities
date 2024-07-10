import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, add_constant, adfuller

nInst = 50
currentPos = np.zeros(nInst)

# Cache
pairs_cache = defaultdict(None)
beta_matrix = None

# Config
window_1 = 5
window_2 = 60

# rolling_window = max(window_1, window_2)
rolling_window = 100

bollinger_window = 20
bollinger_window = min(bollinger_window, rolling_window)
bollinger_margin = 1.5

max_cash_limit = 2_000

zscore_action_threshold = 2
zscore_spread_threshold = 1.5
zscore_neutral_threshold = 0.75
zscore_close_threshold = 0.50

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

def compute_zscore(spread: pd.Series):
    return (spread - spread.mean()) / spread.std()

def bollinger_bands(zscores: pd.Series):
    # z_ma = zscores.rolling(bollinger_window).mean()
    z_ma = zscores.ewm(span=bollinger_window, adjust=False).mean()
    # z_std = zscores.rolling(bollinger_window).std()
    z_std = zscores.ewm(span=bollinger_window, adjust=False).std()
    lower_band = z_ma - bollinger_margin * z_std
    upper_band = z_ma + bollinger_margin * z_std
    return lower_band.iloc[-1], upper_band.iloc[-1]

def find_cointegrated_pairs(df: pd.DataFrame):
    nInst = df.shape[1]
    score_matrix = np.zeros((nInst, nInst))
    beta_matrix = np.ones((nInst, nInst))
    pvalue_matrix = np.ones((nInst, nInst))

    pairs_combi = combinations(df.columns, 2)

    pairs = []
    for S1, S2 in pairs_combi:
        # pvalue, score = coint_simple(df[S1], df[S2])
        # score_matrix[S1, S2] = score

        pvalue, beta = coint_spread(df[S1], df[S2])
        beta_matrix[S1, S2] = beta

        pvalue_matrix[S1, S2] = pvalue
        if pvalue < 0.05:
            pairs.append((S1, S2))

    return pairs, score_matrix, beta_matrix

def trade(S1: pd.Series, S2: pd.Series, beta: np.float64 = 1):
    # if window1 == 0 or window2 == 0:
    #     return 0, 0, 0  # Return neutral positions if windows are incorrectly set.

    # ratios = S1 / S2
    # ma1 = ratios.rolling(window=window1, center=False).mean()
    # ma2 = ratios.rolling(window=window2, center=False).mean()
    # std = ratios.rolling(window=window2, center=False).std()
    # zscore = (ma1 - ma2) / std
    # zscore = zscore.iloc[-1]

    # diffs = np.abs(S1 - S2)
    # mean = diffs.mean()
    # std = diffs.std()
    # curr = diffs.iloc[-1]
    # zscore = np.abs(curr - mean) / std

    # Compute the z-score of the spread
    spread = S1 - beta * S2
    zscores = compute_zscore(spread)
    zscore = zscores.iloc[-1]

    # Bollinger band on current z-score
    lower_band, upper_band = bollinger_bands(zscores)
    # print(f"{lower_band:.2f}, {upper_band:.2f}")

    # Latest price
    latest_price_s1 = S1.iloc[-1]
    latest_price_s2 = S2.iloc[-1]

    # Calculate maximum allowable positions based on $10,000 limit and current prices
    max_pos_S1 = max_cash_limit / latest_price_s1
    max_pos_S2 = max_cash_limit / latest_price_s2

    # Initialize positions
    countS1 = currentPos[S1.name]
    countS2 = currentPos[S2.name]

    # Determine positions based on bollinger bands
    if zscore >= upper_band:            # Spread is too large, S1 > S2 == short S1, long S2
        countS1 = -int(max_pos_S1)
        countS2 = int(max_pos_S2)
    elif zscore <= lower_band:          # Spread is too small, S1 < S2 == long S1, short S2
        countS1 = int(max_pos_S1)
        countS2 = -int(max_pos_S2)

    # elif np.abs(zscore) <= zscore_neutral_threshold:    # Neutral zone, halve positions
    #     countS1 = min(countS1 / 2, max_pos_S1)
    #     countS2 = min(countS2 / 2, max_pos_S2)
    # elif np.abs(zscore) <= zscore_close_threshold:      # Close zone, close positions
    #     countS1 = 0
    #     countS2 = 0

    return countS1, countS2, zscore, lower_band, upper_band

def getMyPosition(prcSoFar):
    global pairs_cache, beta_matrix

    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    if not pairs_cache:
        pairs_cache, _, beta_matrix = find_cointegrated_pairs(data)
        print(len(pairs_cache))

    # for (stock_1, stock_2) in pairs_cache:
    for (stock_1, stock_2) in [pairs_cache[5]]:
    # for (stock_1, stock_2) in [(0, 1)]:
        # print(stock_1, stock_2)
        S1, S2 = data[stock_1].iloc[-rolling_window:], data[stock_2].iloc[-rolling_window:]
        currentPos[stock_1], currentPos[stock_2], zscore, lower_band, upper_band = trade(S1, S2, beta_matrix[stock_1, stock_2])

    return currentPos, zscore, lower_band, upper_band
