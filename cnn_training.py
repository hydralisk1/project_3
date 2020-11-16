from pandas_datareader import DataReader
from datetime import timedelta, date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_squared_error
import joblib

def fill_date(df):
  new_df = df.copy()
  for i in range(1, len(df)):
    diff = (df.index[i] - df.index[i-1]).days
    if diff > 1:
      for j in range(1, diff):
        if (df.index[i-1]+timedelta(days=j)).weekday() != 5 and (df.index[i-1]+timedelta(days=j)).weekday() != 6:
          new_df.loc[df.index[i-1]+timedelta(days=j)] = df.iloc[i-1].copy()
  new_df = new_df.sort_index()

  return new_df

def remove_weekend(df):
    switch = True

    while(switch):
        idx = df.index[0]
        if idx.weekday() == 0:
            switch = False
        else:
            print(idx)
            df = df.drop(index=idx)

    switch = True

    while(switch):
        idx = df.tail(1).index[0]
        if idx.weekday() == 4:
            switch = False
        else:
            print(idx)
            df = df.drop(index=idx)

    return df

def training_model(code):
    today = date.today().strftime("%Y-%m-%d")
    df = DataReader(code, data_source="yahoo", start="1980-01-01", end=today)

    new_df = fill_date(df)
    training_data = remove_weekend(new_df)
    training_data = training_data.drop(columns=["High", "Low", "Open", "Volume", "Adj Close"])

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train, y_train = [], []
    
    for i in range(0, len(training_data)-10, 5):
        X_train.append(training_data[i:i+5])
        y_train.append(training_data[i+5:i+10])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(5))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, epochs=200)

    model_name = "static/cnn_models/"+code+"_model.h5"
    scaler_name = "static/cnn_scalers/"+code+"_scaler.gz"

    model.save(model_name)
    joblib.dump(scaler, scaler_name)

training_model("TSLA")
