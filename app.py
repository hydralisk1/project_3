from flask import Flask, render_template, request
import pandas_datareader as web
import pandas as pd
from datetime import date, timedelta
from whatwehave import what_we_have
from read_data import read_predcited_data

app = Flask(__name__)
companies = what_we_have()

@app.route("/")
def index():
    return render_template("index.html", companies=companies)

@app.route("/chart", methods=["POST"])
def chart():
    code = request.form["stock"]
    if date.today().weekday() == 5 or date.today().weekday() == 6:
        end = date.today() - timedelta(days=(date.today().weekday()-4))
    else:
        end = date.today() - timedelta(days=date.today().weekday()+1)
    end = end.strftime("%Y-%m-%d")
    start="2012-01-01"
    df = web.DataReader(code, data_source="yahoo", start=start, end=end)
    df = df.reset_index()
    df.Date = df.Date.dt.strftime("%Y-%m-%d")
    value = [{"date": df.loc[i, "Date"],
              "high": float(df.loc[i, "High"]),
              "low": float(df.loc[i, "Low"]),
              "open": float(df.loc[i, "Open"]),
              "close": float(df.loc[i, "Close"]),
              "volume": float(df.loc[i, "Volume"]),
              "adj_close": float(df.loc[i, "Adj Close"])} for i in df.index]
    pred_prices = [str(f) for f in read_predcited_data(code)]
    
    return render_template("chart.html", value=value, companies=companies, code=code, pred_prices=pred_prices)

if __name__ == "__main__":
    app.run(debug=True)