import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas_datareader as web
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.backend import clear_session
from datetime import date
import joblib

def training_model(code):
    today = date.today().strftime("%Y-%m-%d")
    df = web.DataReader(code, data_source="yahoo", start="1980-01-01", end=today)

    training_data = df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train, y_train = [], []
    
    for i in range(60, len(training_data)-6):
        X_train.append(training_data[i-60:i])
        y_train.append(training_data[i:i+7])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=200, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(7))
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, epochs=100)

    model_name = "static/models/"+code+"_model.h5"
    scaler_name = "static/scalers/"+code+"_scaler.gz"

    model.save(model_name)
    joblib.dump(scaler, scaler_name)

companies = ["F", "AAL", "NOK", "PLUG", "FCEL", "ACB", "PFE", "VALE", "BABA", "SPAQ", "INTC", "AMD", "WFC", "PDD", "XOM",
"APWC", "T", "PBR", "ABEV", "M", "GNUS", "NCLH", "PCG", "MRO", "JD", "TTNP", "DIS", "DAL", "AMC", "KNDI", "RIG", "SIRI", "FCX", "UAL", "BA", "GPOR",
"BSX", "SNAP", "C"]

#companies = ["NFLX", "AAPL", "MSFT", "CSCO", "BAC", "GOOGL", "ZM"]

for cm in companies:
    clear_session()
    training_model(cm)
