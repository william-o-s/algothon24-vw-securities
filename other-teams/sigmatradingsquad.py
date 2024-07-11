
import numpy as np
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
lastAction = np.zeros(nInst)
priceBoughtAt = np.zeros(nInst)
hardStopPercent = 0.03


def stopLoss():

    return

# checks the price after planned to buy/sell and current position
#  if less than the hardStop price then return the planend amount to buy /sell
# else return 0 to buy/sell
#  We can edit this to buy till it reaches maximum price by adding an equation
def checkHardPtStop(currPrice, numToBuySell, stockNum):
    global hardStopPercent
    global currentPos
    # Currently rounding the hard stop price by using int
    # Change to a float, double, etc, if we want more precision

 
    hardStopPrice = int(10000 * hardStopPercent)

    totalBuy = numToBuySell * currPrice + currentPos[stockNum] *currPrice
    # print("stockNum is " + str(stockNum))
    # print("currPrice is " + str(currPrice))
    # print("numToBuySell is " + str(numToBuySell))
    # print("hardStopPrice is " + str(hardStopPrice))
    # print("totalBuy is " + str(totalBuy) + " numToBuySell * currPrice = " + str(numToBuySell * currPrice) + " currentPos[stockNum] *currPrice = " + str(currentPos[stockNum] *currPrice))

    if (totalBuy > hardStopPrice or (-totalBuy) < hardStopPrice):
        return 0
    else:
        return numToBuySell


def getMyPosition(prcSoFar):
    global currentPos
    global lastAction
    # rows are days 
    # columns are the stocks
    days, stocks = prcSoFar.shape
    # (nins, nt) = prcSoFar.shape

    # print(type(prcSoFar))
    for col in range(0, 50):
        # print("Stock number " + str(col))
        stock_df = pd.DataFrame({'Close': prcSoFar[col, :]})
        stock_df['ema_short'] = stock_df['Close'].ewm(span=20, adjust=False).mean()
        stock_df['ema_long'] = stock_df['Close'].ewm(span=50, adjust=False).mean()
        stock_df['bullish'] = 0.0
        stock_df['bullish'] = np.where(stock_df['ema_short'] > stock_df['ema_long'], 1.0, 0.0)
        stock_df['crossover'] = stock_df['bullish'].diff()
        # print(stock_df)
        # Check the last element in 'crossover'
        last_crossover_value = stock_df['crossover'].iloc[-1]
        currPrice = stock_df['Close'].iloc[-1]

        # buy signal == 1
        if last_crossover_value == 1:
            numStocksBuySell = checkHardPtStop(currPrice, 100, col)
            currentPos[col] = numStocksBuySell
            
            if (currentPos[col] == 0):
                lastAction[col] = 1
            else:
                lastAction[col] = 0
            # retVal = call helper function
            # currentPos[col] = +-retVal
        # sell signal == -1
        elif last_crossover_value == -1:
            # print("sell signal")
            numStocksBuySell = checkHardPtStop(currPrice, -100, col)
            currentPos[col] = numStocksBuySell
            if (currentPos[col] == 0):
                lastAction[col] = -1
            else:
                lastAction[col] = 0
        else:
            currentPos[col] = 0
    # print("currentPos")
    # print(currentPos)
    return currentPos
