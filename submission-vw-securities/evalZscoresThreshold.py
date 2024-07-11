#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pairs_trading_threshold import getMyPosition as getPosition

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

        # Clip position to max based on current prices
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)

        # Calculate dollar volume traded based on clipped position change
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume

        # Calculate wealth gained/lost
        # NOTE: clipping of position also incurs commission
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)

        # Calculate daily PnL
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

# Toggle
window_start = 0
window_end = 250
zscore_action_threshold = 1
zscore_neutral_threshold = 0.25

window_end += 1     # Include window end date
plot_window_index = np.arange(500+window_start, 500+window_end, 1)

pll = pll[window_start:window_end]
zscores = zscores[window_start:window_end]

# Plot pll
fig, ax1 = plt.subplots(figsize=(20,5))
ax1.plot(plot_window_index, pll, label='$')
ax1.axhline(meanpl, color='black', label='P&L mean')

ax2 = ax1.twinx()
ax2.axhline(zscore_action_threshold, label='Action', color='violet', linestyle='--')
ax2.axhline(zscore_neutral_threshold, label='Neutral', color='pink', linestyle=':')
ax2.plot(plot_window_index, zscores, label='Z', color='red')

plt.ylabel('$')
plt.title('Daily P&L')
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig('daily pll.png')
plt.close(fig)

# Plot cumulative pll
fig, ax1 = plt.subplots(figsize=(20,5))
ax1.plot(plot_window_index, np.cumsum(pll))

ax2 = ax1.twinx()
ax2.axhline(zscore_action_threshold, label='Action', color='violet', linestyle='--')
ax2.axhline(zscore_neutral_threshold, label='Neutral', color='pink', linestyle=':')
ax2.plot(plot_window_index, zscores, label='Z', color='red')

plt.ylabel('$')
plt.title('Cumulative P&L')
plt.legend(fancybox=True, framealpha=0.5)
plt.savefig('cumulative pll.png')
plt.close(fig)
