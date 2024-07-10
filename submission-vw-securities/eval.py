#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pairs_trading import getMyPosition as getPosition

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


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    zscores = []
    (_, nt) = prcHist.shape
    for t in range(500, 751):   # can change to 500 to reflect current
        prcHistSoFar = prcHist[:, :t]
        newPosOrig, zscore = getPosition(prcHistSoFar)

        zscores.append(zscore)

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
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    zscores = np.array(zscores)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (pll, zscores, plmu, ret, plstd, annSharpe, totDVolume)


(pll, zscores, meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)

# Plot pll
fig, ax1 = plt.subplots(figsize=(20,5))
ax1.plot(pll, label='$')
ax1.axhline(meanpl, color='black', label='P&L mean')
ax2 = ax1.twinx()
ax2.plot(zscores, label='Z', color='red')
ax2.axhline(1, color='red', linestyle='--')
ax2.axhline(0.75, color='green', linestyle=':')
plt.title('Strategy P&L')
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig('strategy.png')
plt.close(fig)

# Plot cumulative pll
fig, ax1 = plt.subplots(figsize=(20,5))
ax1.plot(np.cumsum(pll))
ax2 = ax1.twinx()
ax2.plot(zscores, label='Z', color='red')
ax2.axhline(1, color='red', linestyle='--')
ax2.axhline(0.75, color='green', linestyle=':')
plt.ylabel('$')
plt.title('Cumulative P&L')
plt.savefig('cumulative.png')
plt.close(fig)
