import pandas_datareader as web
from datetime import date, timedelta
from tensorflow.keras.models import load_model
import joblib
import numpy as np

def prediction(code):
    end = date.today().strftime("%Y-%m-%d")
    start = date.today() - timedelta(days=180)
    start = start.strftime("%Y-%m-%d")

    df = web.DataReader(code, data_source="yahoo", start=start, end=end)
    new_df = df.tail(60)
    data = new_df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)

    scaler = joblib.load("static/scalers/"+code+"_scaler.gz")
    model = load_model("static/models/"+code+"_model.h5")

    data = np.array([scaler.transform(data)])
    print(data.shape)
    pred = model.predict(data)

    pred = scaler.inverse_transform(pred)
    
    return list(pred[0, :])

def cnn_prediction(code):
    day_today = date.today().weekday()
    if day_today == 5 or day_today == 6:
        start = date.today() - timedelta(days=day_today)
    else:
        start = date.today() - timedelta(days=(day_today+7))
    
    end = start + timedelta(days=4)
    
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    df = web.DataReader(code, data_source="yahoo", start=start_date, end=end_date)
    
    data = df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)

    scaler = joblib.load("static/cnn_scalers/"+code+"_scaler.gz")
    model = load_model("static/cnn_models/"+code+"_model.h5")

    data = np.array([scaler.transform(data)])
    
    pred = model.predict(data)
    pred = scaler.inverse_transform(pred)
    
    return pred.flatten()
