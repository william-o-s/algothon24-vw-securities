import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import permutations
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, add_constant, adfuller

nInst = 50
currentPos = np.zeros(nInst)

beta_cache = defaultdict(None)

# Config
rolling_window = 50
bollinger_window = 50
bollinger_window = min(bollinger_window, rolling_window)

max_cash_limit = 10000

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
    return beta

def compute_zscore(spread: pd.Series):
    return (spread - spread.mean()) / spread.std()

def bollinger_bands(zscores: pd.Series):
    z_ma = zscores.rolling(bollinger_window).mean()
    z_std = zscores.rolling(bollinger_window).std()
    upper_band = z_ma + 2 * z_std
    lower_band = z_ma - 2 * z_std
    return lower_band.iloc[-1], upper_band.iloc[-1]

def trade(S1: pd.Series, S2: pd.Series, beta: np.float64):
    # Compute the z-score of the spread
    spread = S1 - beta * S2
    zscores = compute_zscore(spread)
    zscore = zscores.iloc[-1]

    # Bollinger band on current z-score
    lower_band, upper_band = bollinger_bands(zscores)

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
        countS1 = int(max_pos_S1)
        countS2 = -int(max_pos_S2)
    elif zscore <= lower_band:          # Spread is too small, S1 < S2 == long S1, short S2
        countS1 = -int(max_pos_S1)
        countS2 = int(max_pos_S2)

    return countS1, countS2

def getMyPosition(prcSoFar, stock_1, stock_2, reset):
    global currentPos

    if reset:
        currentPos = np.zeros_like(currentPos)

    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    if (stock_1, stock_2) not in beta_cache:
        beta_cache[(stock_1, stock_2)] = coint_spread(data.iloc[:500, stock_1], data.iloc[:500, stock_2])

    S1, S2 = data.iloc[-rolling_window:, stock_1], data.iloc[-rolling_window:, stock_2]
    currentPos[stock_1], currentPos[stock_2] = trade(S1, S2, beta_cache[(stock_1, stock_2)])

    return currentPos
