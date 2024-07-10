import numpy as np
import pandas as pd
import statsmodels.api as sm

from collections import defaultdict

nInst = 50
currentPos = np.zeros(nInst)

model_cache = defaultdict(None)

def train_mlr_model(df: pd.DataFrame, stock: int):
    # MLR model -> y_{t+1} = x_t * beta + e
    # Uses previous values of all 50 stocks to predict next day's price

    Y_train = df[stock].shift(-1).dropna()

    X_train = df.iloc[:-1, df.columns]
    X_train = sm.add_constant(X_train)

    model_cache[stock] = sm.OLS(endog=Y_train, exog=X_train).fit()

def predict(df: pd.DataFrame, stock: int):
    current_prices = df.iloc[-1]
    current_prices = sm.add_constant([current_prices], has_constant='add')

    stock_model = model_cache[stock]
    return stock_model.predict(exog=current_prices)

def getMyPosition(prcSoFar):
    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    for stock in data.columns:
        if stock not in model_cache:
            train_mlr_model(data, stock)

        latest_price = data.iloc[-1, stock]
        predicted_price = predict(data, stock)
        currentPos[stock] = np.sign(predicted_price - latest_price) * int(10_000 / latest_price) * 0.2

    return currentPos