from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from databases.nnemo_db import Base, NNTraining
from emotion_detection.lbp_emotion_detection import LBPEmotionDetection


NN_EMOT_DB = 'sqlite:///../../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../../databases/img/'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

nntraining = session.query(NNTraining).first()
print(nntraining.info)
ed = LBPEmotionDetection(engine)
ed.load_network(nntraining)
print(ed.nn.learningrate)