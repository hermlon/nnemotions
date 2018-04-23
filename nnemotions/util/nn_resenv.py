from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base


class NNResEnv:

    def __init__(self, db_file, model_dir, img_dir):
        # init database and provide session
        engine = create_engine(db_file)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        self.db = DBSession()
        self.model_dir = model_dir
        self.img_dir = img_dir