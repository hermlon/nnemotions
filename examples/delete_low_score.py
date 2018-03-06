from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, NNTraining
import os


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_MODEL_DIR = '../../databases/nn_models'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

low_scores = session.query(NNTraining).filter(NNTraining.score < 80.0).all()

for low_score in low_scores:
    print('Removing %s with score %.2f%%' % (low_score.nn_saved_name, low_score.score))
    file = os.path.join(NN_MODEL_DIR, low_score.nn_saved_name)
    session.delete(low_score)
    try:
        os.remove(file)
    except FileNotFoundError:
        print('Did not find on disk: %s' % file)
session.commit()
