
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    # Short and hold
    curPrices = prcSoFar[:, -1]
    return np.array([-int(x) for x in 10_000 / curPrices])
