import pandas_datareader as web
from datetime import date, timedelta
from tensorflow.keras.models import load_model
import joblib
import numpy as np

def test_prediction(code):
    end = date.today().strftime("%Y-%m-%d")
    start = date.today() - timedelta(days=180)
    start = start.strftime("%Y-%m-%d")

    df = web.DataReader(code, data_source="yahoo", start=start, end=end)

    new_df = df.tail(67).drop(df.tail(7).index)
    data = new_df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)

    scaler = joblib.load("static/scalers/"+code+"_scaler.gz")
    model = load_model("static/models/"+code+"_model.h5")

    data = np.array([scaler.transform(data)])
    print(data.shape)
    pred = model.predict(data)

    pred = scaler.inverse_transform(pred)
    
    return list(pred[0, :])