import numpy as np
import pandas as pd
from mean_reversion_tuning import getMyPosition as getPosition

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

def calcPL(prcHist, n1, base_offset):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(500, 750):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar, n1, base_offset)
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
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, plstd, annSharpe, totDVolume)

def tune_parameters(prcAll):
    best_score = -np.inf
    best_params = {}
    for n1 in range(20, 51, 10):  # Testing different window sizes from 20 to 50
        for base_offset in np.arange(0.005, 0.021, 0.005):  # Testing offsets from 0.005 to 0.02
            meanpl, plstd, sharpe, dvol = calcPL(prcAll, n1, base_offset)
            score = meanpl - 0.1 * plstd
            print(f"n1: {n1}, base_offset: {base_offset}, Score: {score}")
            if score > best_score:
                best_score = score
                best_params = {'n1': n1, 'base_offset': base_offset, 'score': score}
    return best_params

best_params = tune_parameters(prcAll)
print("Best Parameters:")
print("n1: ", best_params['n1'])
print("base_offset: ", best_params['base_offset'])
print("Best Score: ", best_params['score'])
