import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.linear_model import LinearRegression

nInst = 50
currentPos = np.zeros(nInst)

model_cache = defaultdict(None)

def train_mlr_model(df: pd.DataFrame, stock: int):
    X_train = df.loc[:, df.columns != stock]
    Y_train = df[stock]

    # build model
    model_cache[stock] = LinearRegression(fit_intercept=False).fit(X_train, Y_train)

def predict(df: pd.DataFrame, stock: int):
    stock_model = model_cache[stock]
    features = df.iloc[-1, df.columns != stock]
    features = pd.DataFrame([features.tolist()], columns=features.index)

    return stock_model.predict(features)

def getMyPosition(prcSoFar):
    data = pd.DataFrame(prcSoFar.T)
    (nt, nInst) = prcSoFar.shape

    if not model_cache:
        for stock in data.columns:
            train_mlr_model(data, stock)

    for stock in data.columns:
        latest_price = data.iloc[-1, stock]
        predicted_price = predict(data, stock)
        currentPos[stock] = np.sign(predicted_price - latest_price)

    return currentPos