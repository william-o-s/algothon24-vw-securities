import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
clusters = [[41,35]]#[[19, 32, 48], [22, 46, 43, 37], [41, 35], [20, 25, 26]]  # clusters from analysis
#left over,[14,42,0.765778,50],[1,10,1.147285,20],[43,9,0.234708,60],[7,49,-1.214857,60],[28,45,-0.143413,100
#neg[20,35,0.51603,20],[43,9,0.234708,60],[7,49,-1.214857,50]
#pos [[14,42,0.765778,50],[28,45,-0.143413,100]]
coint_pairs_hedge = [[14,42,0.765778,50],[28,45,-0.143413,100]] #
def calculate_bollinger_bands(prices, window=20, num_std_dev=1):
    sma = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return sma, upper_band, lower_band

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_returns(prices):
    returns = prices.pct_change()
    return returns

def calculate_volatility(prices, window=14):
    return prices.pct_change().rolling(window=window).std()
#hedge is -0.45987
def trade_spread(prices, i, j, hedge_ratio,window,k,units,macy,macx):
    df = pd.DataFrame(prices.T, columns=[f'Instrument_{i}' for i in range(50)])
    hedge = pd.DataFrame({'hedge': [hedge_ratio]*(df.shape[0])})
    pairs_df = df[[f"Instrument_{i}", f"Instrument_{j}"]].copy()
    Y = df[f"Instrument_{i}"]
    X = df[f"Instrument_{j}"]
    spread = Y - hedge_ratio*X
    #print(spread)
    pairs_df['spread'] = spread
    #plt.plot(spread)
    #plt.show()
    #print(f"spread is {pairs_df['spread']}")
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    #print(f"rolling_std is {rolling_std}")
    upper_band = rolling_mean + k*rolling_std
    lower_band = rolling_mean - k*rolling_std
    latest_upper_band = upper_band.iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    #print(f"rolling_mean is {rolling_mean}")
    latest_spread = pairs_df['spread'].iloc[-1]
    #print(f"latest spread is {latest_spread},Upper: {latest_upper_band}, Lower: {latest_lower_band}")
    pos_x  =0 
    pos_y = 0
    if latest_spread > latest_upper_band: #and macx > 0: #stock price will revert down, sell Y and buy X
        pos_y = -units
        pos_x = -hedge_ratio*units
        #print("reverting down")
    elif latest_spread < latest_lower_band :  # price will revert up, spread will increaase
        pos_y = units
        pos_x = hedge_ratio*units
       # print("reverting up")
    return pos_y, pos_x



def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)
    # Calculate indicators
    macd, _ = calculate_macd(pd.DataFrame(prices.T))
    rsi = calculate_rsi(pd.DataFrame(prices.T))
    sma, upper_band, lower_band = calculate_bollinger_bands(pd.DataFrame(prices.T))
    volatility = calculate_volatility(pd.DataFrame(prices.T))
    for item in coint_pairs_hedge:
        pair_1 = item[0]
        pair_2 = item[1]
        hedge_rat = item[2]
        windowpair = item[3]
        a_macd = macd.iloc[:,pair_1].iloc[-1]
        b_macd = macd.iloc[:,pair_2].iloc[-1]
        #print((a_macd,b_macd))
        pos_a,pos_b = trade_spread(prices,pair_1,pair_2,hedge_rat,windowpair,1,100,a_macd,b_macd)
        if (pos_a == pos_b == 0):
        # print("No Change, Looking back")
            new_prices = prices[:, :-1]
            new_macd = macd.iloc[:-1,:]
            while (new_prices.shape[1] > windowpair):
               # print(f"length is {new_prices.shape[1]}")
                a_macd = new_macd.iloc[:,pair_1].iloc[-1]
                b_macd = new_macd.iloc[:,pair_2].iloc[-1]
                pos_a, pos_b =  trade_spread(new_prices,pair_1,pair_2,hedge_rat,windowpair,1,100,a_macd,b_macd)
                new_prices = new_prices[:, :-1]
                new_macd = new_macd.iloc[:-1,:]
                if (pos_a != pos_b):
                    break 
        positions[pair_1] = pos_a
        positions[pair_2] = pos_b

        # print(f"position of {pair_1} is {positions[pair_1]}, position of {pair_2} is {positions[pair_2]}")
    #SPREAD TRADING STRATEGY FOR 7 & 49


    # # Use provided clusters to determine positions
    # for cluster in clusters:
    #     cluster_macd = macd.iloc[-1, cluster].mean()
    #     cluster_rsi = rsi.iloc[-1, cluster].mean()
    #     cluster_volatility = volatility.iloc[-1, cluster].mean()
    #     cluster_upper_band = upper_band.iloc[-1, cluster].mean()
    #     cluster_lower_band = lower_band.iloc[-1, cluster].mean()
        
    #     # Define position sizing based on volatility (inverse)
    #     #position_size = 10000 / (cluster_volatility + 1e-6)  # Adding small value to avoid division by zero
        
    #     # Advanced strategy: Use multiple indicators and dynamic position sizing
    #     for stock in cluster:
    #         current_price = prices[stock, -1]
    #         if current_price < cluster_lower_band and cluster_macd > 0 and cluster_rsi < 70:
    #             for stock in cluster:
    #                 positions[stock] = position_size / prices[stock, -1]  # Long position
    #         elif current_price > cluster_upper_band and cluster_macd < 0 and cluster_rsi > 30:
    #             for stock in cluster:
    #                 positions[stock] = -position_size / prices[stock, -1]  # Short position
    # pos_7,pos_49 = trade_spread(prices,7,49,-0.45987,50,1.5,100)
    # if (pos_7 == pos_49 == 0):
    #    # print("No Change, Looking back")
    #     new_prices = prices[:, :-1]
    #    # print(f"length is {new_prices.shape[1]}")
    #     while (new_prices.shape[1] > 50):
    #         pos_7, pos_49 = trade_spread(new_prices,7,49,-0.45987,50,1.5,100)
    #         new_prices = new_prices[:, :-1]
    #         if (pos_7 != pos_49):
    #             break 
    # positions[7] = pos_7
    # positions[49] = pos_49
    # print(f"position_7 = {pos_7}, position_49 = {pos_49}")
    # Ensure position limits
    for i in range(nInst):
        current_price = prices[i, -1]
        position_value = positions[i] * current_price
        if abs(position_value) > 10000:
            positions[i] = 10000 / current_price if position_value > 0 else -10000 / current_price
    
    return positions.astype(int)