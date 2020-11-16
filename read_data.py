from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# DB URI
uri = "postgres://postgres:hy046790hy@localhost:5432/project_3"

def read_predcited_data(code):
    engine = create_engine(uri)
    Base = automap_base()
    Base.prepare(engine, reflect = True)

    Predicted = Base.classes.predicted
    
    session = Session(engine)

    query = session.query(Predicted.mon, Predicted.tue, Predicted.wed, Predicted.thu, Predicted.fri).filter(Predicted.code == code).order_by(Predicted.week_start_date.desc())

    data = []
    # Run queries to load data from the database server
    for row in query:
        for pred in row:
            data.append(pred)

    return data