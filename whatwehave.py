from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

def what_we_have():
    uri = "postgres://postgres:hy046790hy@localhost:5432/project_3"
    engine = create_engine(uri)
    Base = automap_base()
    Base.prepare(engine, reflect = True)

    Predicted = Base.classes.predicted
    
    session = Session(engine)

    query = session.query(Predicted.code)

    data = []
    # Run queries to load data from the database server
    for row in query:
        data.append(row.code)
        
    return data