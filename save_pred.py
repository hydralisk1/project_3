from glob import glob
from predicting import cnn_prediction
from sqlalchemy import create_engine, Table, Column, String, Integer, MetaData, Date, Float
from datetime import date

uri = "postgres://postgres:hy046790hy@localhost:5432/project_3"
db = create_engine(uri)
meta = MetaData(db)
predict_table = Table("predicted", meta,
                      Column("id", Integer, primary_key=True, autoincrement=True),
                      Column("code", String),
                      Column("mon", Float),
                      Column("tue", Float),
                      Column("wed", Float),
                      Column("thu", Float),
                      Column("fri", Float),
                      Column("week_start_date", Date))

with db.connect() as conn:
    code = "ZM"
    pred = cnn_prediction(code)
    data = {
        "code": code,
        "mon": float(pred[0]),
        "tue": float(pred[1]),
        "wed": float(pred[2]),
        "thu": float(pred[3]),
        "fri": float(pred[4]),
        "week_start_date": date.today()
    }

    conn.execute(predict_table.insert(), data)
