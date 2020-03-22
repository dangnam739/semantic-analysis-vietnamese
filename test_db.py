# -*- python-mode -*-
# -*- coding: utf-8 -*-
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

os.environ["DATABASE_URL"] = "postgresql://postgres:123456@127.0.0.1:5432/sav"

# DATABASE_URL = "postgresql://postgres:123456@127.0.0.1:5432/sav"

engine = create_engine(os.getenv("DATABASE_URL"))
db = scoped_session(sessionmaker(bind=engine))

content = "Quán này rất ngon."
label = 1

db.execute("INSERT INTO labeled_paragraph (content, label) VALUES (:content, :label)",
            {"content": content , "label": label})
db.commit()