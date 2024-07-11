import numpy as np

# number of instruments
nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    
    (nins, nt) = prcSoFar.shape
    
    if nt < 50:
        return np.zeros(nins)
    
    # Calculate difference in prices between last day and day 50 days ago
    # the -1 is the most recent and we subtract the oldest 1 row
    price_diff = prcSoFar[:, -1] - prcSoFar[:, -300]

    # Initialize adjustment array
    adjustment = np.zeros(nins)

    # Adjust positions based on price difference
    for i in range(nins):
        # some random risk calculation i made up idk 
        # if this is legit actually the more i think the more idk ah well if it works it works
        #  if the change in price recently is more than average, then there is higher risk
        #  > 1 = higher that usual
        #  < 1 = lower than usual
        sum_of_differences = 0
        for j in range(50, nt):
            sum_of_differences += abs(prcSoFar[i,j] - prcSoFar[i,j - 50])
        average_of_differences = sum_of_differences/(nt / 50)
        current_difference = abs(prcSoFar[i,-1] - prcSoFar[i,-50])
        risk = 0
        if average_of_differences != 0 and current_difference != 0:
            # if current difference is more than usual, then i want to take a lower risk
            # hence why i inverse? it (i divide 1 by it)
            # I want to limit risk to 0-1(see below)
            # the range rn of current difference/average difference is 0-infinity
            # so its 1/0 = 1 to 1/ infinity = 0
            risk = 1 / (current_difference / average_of_differences)
        
        if price_diff[i] > 0:
            # you get 10,000 total per stock
            # so each stock is at most 10000
            # so that means that 10,000/prcSoFar[i,-1] = the maximum amount of stock i can buy
            # which means I want to limit risk to a number between 0-1
            adjustment[i] = (10000/ prcSoFar[i,-1]) * risk
        elif price_diff[i] < 0 and current_difference != 0:
            adjustment[i] = (-10000 / prcSoFar[i, -1]) * risk
        else:
            adjustment[i] = 0
    
    # Update current positions directly
    currentPos = adjustment
    
    # print(currentPos) 
    return currentPos