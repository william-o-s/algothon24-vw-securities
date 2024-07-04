import pandas as pd
import numpy as np

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

nInst = 50
currentPos = np.zeros(nInst)

# Input
days_shifted = 1

def sample(prcSoFar: np.ndarray):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

def get_clusters(df: pd.DataFrame) -> tuple:

    ### Cluster based on daily movements
    daily_movements = df.T.apply(lambda col: col - col.shift(days_shifted), axis=0)
    daily_movements.dropna(axis=0, inplace=True)
    daily_movements = daily_movements.T
    print(daily_movements.shape)

    normalizer = Normalizer()
    clustering_model = KMeans(n_clusters=4, init='random' , n_init=5, max_iter = 100_000)
    pipeline = make_pipeline(normalizer, clustering_model)
    pipeline.fit(daily_movements)
    labels = pipeline.predict(daily_movements)

    ### View cluster with daily returns and price
    cluster_labels = pd.DataFrame(labels, columns=['Cluster'])
    # Assuming stock_symbols is a list of stock names
    stock_symbols = [f"stock{i}" for i in range(50)]

    # Set the stock symbols as an index to the cluster_labels DataFrame
    cluster_labels['Stock'] = stock_symbols
    cluster_labels.set_index('Stock', inplace=True)

    daily_stock_price_transposed = df.T
    daily_return_with_cluster = daily_movements.join(cluster_labels)
    daily_price_with_cluster = daily_stock_price_transposed.join(cluster_labels)

    return daily_return_with_cluster, daily_price_with_cluster

def calculateFairPrice(daily_return_with_cluster, daily_price_with_cluster, stockname, alpha = 0.1):
    # Ensure the stock name is in the DataFrame
    if stockname not in daily_price_with_cluster.index:
        raise ValueError("Stock name not found in the data")

    # Calculate the average historic price for the stock
    avg_historic_price = daily_price_with_cluster.loc[stockname, :].drop('Cluster').astype(float).mean()
    
    # Get the cluster of the stock
    cluster = daily_price_with_cluster.at[stockname, 'Cluster']
    
    # Calculate the average return rate for the cluster
    # Get all stocks in the same cluster except the current stock
    cluster_stocks = daily_price_with_cluster[daily_price_with_cluster['Cluster'] == cluster].index
    cluster_stocks = cluster_stocks.drop(stockname)  # Exclude the current stock

    # Filter the returns DataFrame for the current cluster and select only the last day
    # Assuming the last day is consistently the last column before 'Cluster'
    last_day_returns = daily_return_with_cluster.loc[cluster_stocks].drop('Cluster', axis=1).iloc[:, -1].astype(float)
    avg_cluster_return = last_day_returns.mean()  # Average of last day returns only, excluding the current stock

    # Calculate the volatility of the stock
    # Volatility is the standard deviation of the stock's returns
    stock_volatility = daily_return_with_cluster.loc[stockname].drop('Cluster').astype(float).std()

    # Calculate the fair price using the given formula
    # Assuming positive volatility adjustment; adjust the sign as necessary
    fair_price_up = avg_historic_price * ((1 + avg_cluster_return) + stock_volatility * alpha)
    fair_price_down = avg_historic_price * ((1 + avg_cluster_return) - stock_volatility * alpha)
    
    return fair_price_down, fair_price_up

def getLastDayPrice(daily_price_with_cluster, stockname):
    # Ensure the stock name is in the DataFrame
    if stockname not in daily_price_with_cluster.index:
        raise ValueError("Stock name not found in the data")

    # Get the last day's price, excluding the 'Cluster' column if present
    last_day_price = daily_price_with_cluster.loc[stockname].drop('Cluster').iloc[-1]
    
    return last_day_price

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    # read as dataframe
    df = pd.DataFrame(prcSoFar.T, columns=[f"stock{i}" for i in range(50)])

    # get data
    daily_return_with_cluster, daily_price_with_cluster = get_clusters(df)

    # decide position for each stock
    print(calculateFairPrice(daily_return_with_cluster, daily_price_with_cluster, "stock0", 0.5))
    print(getLastDayPrice(daily_price_with_cluster, "stock0"))


    # convert back to ndarray
    return sample(prcSoFar)


# first need to figure the data structure of the input
# and identify the clusters
# and then call the fair price function and compare and return the position
