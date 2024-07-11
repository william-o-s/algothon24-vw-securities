import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

nInst = 50
currentPos = np.zeros(nInst)

df = pd.read_csv('prices.txt', sep='\s+', header=None)
df.columns = [f'Stock_{i}' for i in range(df.shape[1])]

pairs = []

def find_cointegrated_pairs(data):
    n = data.shape[1]
    keys = data.columns

    for i in range(n):
        for j in range(i+1, n):
            S1 = data.iloc[:, i]
            S2 = data.iloc[:, j]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]

            if pvalue < 0.01:
                pairs.append((keys[i], keys[j], pvalue))

find_cointegrated_pairs(df)
sorted_pairs = sorted(pairs, key=lambda x: x[2])

trading_pairs = [(pair[0], pair[1]) for pair in sorted_pairs]

def getMyPosition(prcSoFar):
    global currentPos 
    (nins, nt) = prcSoFar.shape
    currentPos = np.zeros(nins)

    for pair in trading_pairs:
        stock1 = int(pair[0].split('_')[1])
        stock2 = int(pair[1].split('_')[1])
        
        spread = prcSoFar[stock1, :] - prcSoFar[stock2, :]
        zscore = (spread - spread.mean()) / spread.std()

        stock_1_position_size = 0.1 * prcSoFar[stock1, -1] * abs(zscore[-1]) 
        stock_2_position_size = 0.1 * prcSoFar[stock2, -1] * abs(zscore[-1])
        
        if zscore[-1] > 2:
            currentPos[stock1] -= stock_1_position_size
            currentPos[stock2] += stock_2_position_size
        elif zscore[-1] < -2:
            currentPos[stock1] += stock_1_position_size
            currentPos[stock2] -= stock_2_position_size
        else:
            currentPos[stock1] = 0
            currentPos[stock2] = 0

    return currentPos
