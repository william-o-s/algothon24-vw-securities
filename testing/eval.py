#!/usr/bin/env python

import numpy as np
import pandas as pd
from itertools import permutations
from main import getMyPosition as getPosition

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

pairs = permutations(range(50), 2)
max_score = -np.inf
max_pair = None

for (S1, S2) in [(0,2)]:
    reset = True
    def calcPL(prcHist):
        global reset

        cash = 0
        curPos = np.zeros(nInst)
        totDVolume = 0
        totDVolumeSignal = 0
        totDVolumeRandom = 0
        value = 0
        todayPLL = []
        (_, nt) = prcHist.shape
        for t in range(500, 751):
            prcHistSoFar = prcHist[:, :t]
            newPosOrig = getPosition(prcHistSoFar, S1, S2, reset)
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
            todayPLL.append(todayPL)
            value = cash + posValue
            ret = 0.0
            if (totDVolume > 0):
                ret = value / totDVolume
            # print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
            #     (t, value, todayPL, totDVolume, ret))

            reset = False
        pll = np.array(todayPLL)
        (plmu, plstd) = (np.mean(pll), np.std(pll))
        annSharpe = 0.0
        if (plstd > 0):
            annSharpe = np.sqrt(250) * plmu / plstd
        return (plmu, ret, plstd, annSharpe, totDVolume)

    (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
    score = meanpl - 0.1*plstd
    print(f"===== ({S1}, {S2}) =====")
    print("mean(PL): %.1lf" % meanpl)
    print("return: %.5lf" % ret)
    print("StdDev(PL): %.2lf" % plstd)
    print("annSharpe(PL): %.2lf " % sharpe)
    print("totDvolume: %.0lf " % dvol)
    print("Score: %.2lf" % score)

    if score > max_score:
        max_score = score
        max_pair = (S1, S2)

print(max_pair, ":", max_score)
