#!/usr/bin/env python

import numpy as np
import pandas as pd
from EMA_and_mean_reversion import getMyPosition

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, short_window, long_window, lower_percentile, upper_percentile, stop_loss=0.05, take_profit=0.1):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(250, 500):   # can change to 500 to reflect current
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getMyPosition(prcHistSoFar, short_window, long_window, lower_percentile, upper_percentile)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value

        # Implement stop-loss and take-profit
        if todayPL < -stop_loss * value:
            cash += curPos.dot(curPrices)
            curPos = np.zeros(nInst)
            todayPL = cash + posValue - value
        elif todayPL > take_profit * value:
            cash += curPos.dot(curPrices)
            curPos = np.zeros(nInst)
            todayPL = cash + posValue - value

        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)

# Define ranges for thresholds
lower_percentile_range = range(5, 21, 5)  # Adjust as needed
upper_percentile_range = range(80, 96, 5)  # Adjust as needed

# Initialize variables to track the best performance
best_performance = -np.inf
best_short_window = None
best_long_window = None
best_lower_percentile = None
best_upper_percentile = None

# Loop over all combinations of short and long windows and thresholds
for short_window in range(3, 10):
    for long_window in range(80, 110):
        for lower_percentile in lower_percentile_range:
            for upper_percentile in upper_percentile_range:
                (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll, short_window, long_window, lower_percentile, upper_percentile)
                score = meanpl - 0.1 * plstd
                print(f"score: {score}, short: {short_window}, long: {long_window}, lower: {lower_percentile}, upper: {upper_percentile}")
                if score > best_performance and score != 0:
                    best_performance = score
                    best_short_window = short_window
                    best_long_window = long_window
                    best_lower_percentile = lower_percentile
                    best_upper_percentile = upper_percentile

print("=====")
print(f"Best Short Window: {best_short_window}")
print(f"Best Long Window: {best_long_window}")
print(f"Best Lower Percentile: {best_lower_percentile}")
print(f"Best Upper Percentile: {best_upper_percentile}")
print(f"Best Performance (Score): {best_performance:.2f}")
