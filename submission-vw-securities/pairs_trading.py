import logging, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from math import comb, perm
from itertools import combinations, permutations
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, add_constant, adfuller

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

### Algothon ###
nInst = 50
commRate = 0.0010
dlrPosLimit = 10000
currentPos = np.zeros(nInst)
################

### Pairs Trading ###
rolling_window = 20                # Toggle

pairs_cache = None
beta_matrix = None
#####################

### Bollinger Bands ###
bollinger_margin = 1.5              # Toggle
bollinger_neutral = 0.25              # Toggle
bollinger_window = 20               # Toggle

bollinger_window = min(bollinger_window, rolling_window)
pairs_spread_location = defaultdict(int)        # above upper band = +1
                                                # within bands = 0
                                                # below lower band = -1
#######################

### Kelly Criterion ###
# max_cash_limit = 500
max_cash_limit = dlrPosLimit        # Toggle

wealth_size = np.full(shape=nInst, fill_value=max_cash_limit)
wealth_pnl = np.zeros(shape=nInst)
bought_price = np.zeros(shape=nInst)
#######################

zscore_action_threshold = 2         # Toggle
zscore_spread_threshold = 1.5       # Toggle
zscore_neutral_threshold = 0.75     # Toggle
zscore_close_threshold = 0.50       # Toggle

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

def find_cointegrated_pairs(df: pd.DataFrame, significance=0.05):
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

def compute_zscore(spread: pd.Series):
    return (spread - spread.mean()) / spread.std()

def bollinger_bands(zscores: pd.Series):
    # z_ma = zscores.rolling(bollinger_window).mean()
    z_ma = zscores.ewm(span=bollinger_window, adjust=False).mean()
    # z_std = zscores.rolling(bollinger_window).std()
    z_std = zscores.ewm(span=bollinger_window, adjust=False).std()

    lower_band = z_ma - bollinger_margin * z_std
    lower_neutral_band = z_ma - bollinger_neutral * z_std
    upper_neutral_band = z_ma + bollinger_neutral * z_std
    upper_band = z_ma + bollinger_margin * z_std
    return lower_band.iloc[-1], lower_neutral_band.iloc[-1], upper_neutral_band.iloc[-1], upper_band.iloc[-1]

def wealth_limit(stock, edge=0.65, odds=1):
    half_kelly = 0.5 * (edge - ((1 - edge) / odds))
    return wealth_size[stock] * half_kelly

def trade(S1: pd.Series, S2: pd.Series, beta: np.float64 = 1):
    stock1, stock2 = S1.name, S2.name
    # if window1 == 0 or window2 == 0:
    #     return 0, 0, 0  # Return neutral positions if windows are incorrectly set.

    # ratios = S1 / S2
    # ma1 = ratios.rolling(window=window1, center=False).mean()
    # ma2 = ratios.rolling(window=window2, center=False).mean()
    # std = ratios.rolling(window=window2, center=False).std()
    # zscore = (ma1 - ma2) / std
    # zscore = zscore.iloc[-1]

    # Compute the z-score of the spread
    spread = S1 - beta * S2
    zscores = compute_zscore(spread)
    zscore = zscores.iloc[-1]

    # Bollinger band on current z-score
    lower_band, lower_neutral_band, upper_neutral_band, upper_band = bollinger_bands(zscores)

    # Latest price
    latest_price_s1 = S1.iloc[-1]
    latest_price_s2 = S2.iloc[-1]

    # Calculate maximum allowable positions based on $10,000 limit or Kelly criterion
    kelly_pos_S1 = wealth_limit(stock1) / latest_price_s1
    kelly_pos_S2 = wealth_limit(stock2) / latest_price_s2
    # kelly_pos_S1 = dlrPosLimit / latest_price_s1
    # kelly_pos_S2 = dlrPosLimit / latest_price_s2

    rules_pos_S1 = dlrPosLimit / latest_price_s1
    rules_pos_S2 = dlrPosLimit / latest_price_s2

    max_pos_S1 = min(kelly_pos_S1, rules_pos_S1)
    max_pos_S2 = min(kelly_pos_S2, rules_pos_S2)

    # Initialize positions
    countS1 = currentPos[stock1]
    countS2 = currentPos[stock2]

    # Check current pair-spread location
    previous_spread_location = pairs_spread_location[(stock1, stock2)]      # either +1, 0, -1

    if zscore >= upper_band:
        current_spread_location = 1
    elif zscore <= lower_band:
        current_spread_location = -1
    else:
        current_spread_location = 0

    # Determine positions based on location around bollinger bands
    if previous_spread_location < 1 and current_spread_location == 1:           # Moving upwards
        logging.info('moving upwards')
        # if countS1 > 0 and countS2 < 0:
        #     countS1 = countS2 = 0       # Currently long on spread: exit position before reversal
    elif previous_spread_location == 1 and current_spread_location < 1:         # Mean-reverting downwards
        logging.info('mean-reverting downwards')
        countS1 = -int(max_pos_S1)      # Spread is too large, S1 > S2 == short S1, long S2
        countS2 = int(max_pos_S2)
    elif previous_spread_location > -1 and current_spread_location == -1:        # Moving downwards
        logging.info('moving downwards')
        # if countS1 < 0 and countS2 > 0:
        #     countS1 = countS2 = 0       # Currently short on spread: exit position before reversal
    elif previous_spread_location == -1 and current_spread_location > -1:        # Mean-reverting upwards
        logging.info('mean-reverting upwards')
        countS1 = int(max_pos_S1)       # Spread is too small, S1 < S2 == long S1, short S2
        countS2 = -int(max_pos_S2)

    if lower_neutral_band <= zscore <= upper_neutral_band:
        countS1 = countS2 = 0

    pairs_spread_location[(stock1, stock2)] = current_spread_location

    return countS1, countS2, zscore, lower_band, lower_neutral_band, upper_neutral_band, upper_band

#######################################################################################################################

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
        currentPos[stock_1], currentPos[stock_2], zscore, lower_band, lower_neutral_band, upper_neutral_band, upper_band = trade(S1, S2, beta_matrix[stock_1, stock_2])

    # logging.info(currentPos[:10])

    return currentPos, zscore, lower_band, lower_neutral_band, upper_neutral_band, upper_band   # evalZscores.py
    # return currentPos                                 # eval.py

def update_pnl(prcSoFar):
    global wealth_size, wealth_pnl

    # Clip position to max based on current prices
    current_prices = prcSoFar.iloc[-1]
    all_position_limits = np.array([int(x) for x in dlrPosLimit / current_prices])
    clipped_positions = np.clip(currentPos, -all_position_limits, all_position_limits)

    # Calculate dollar volume traded based on clipped position change
    clipped_position_deltas = clipped_positions - currentPos
    volume_deltas = current_prices * np.abs(clipped_position_deltas)

    # Calculate wealth gained/lost
    # NOTE: clipping of position also incurs commission
    clipping_commissions = volume_deltas * commRate
    clipping_costs = current_prices * clipped_position_deltas + clipping_commissions
    position_value = current_prices * currentPos

    # current_wealth = wealth_size.copy()
    # wealth_size -= clipping_costs
    # wealth_pnl = cash + position_value - (previous_cash + previous_position_value)