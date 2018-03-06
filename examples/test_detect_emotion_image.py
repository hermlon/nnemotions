from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, NNTraining
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from nnemotions.detection.face.input import Input
import cv2

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img'
NN_MODEL_DIR = '../../databases/nn_models'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

#nntraining = session.query(NNTraining).filter_by(id=40).first()
nntraining = session.query(NNTraining).first()
print('Score: %s' % nntraining.score)
ed = LBPEmotionDetection(engine, NN_MODEL_DIR)
ed.load_network(nntraining)

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('../../databases/test_img/sad2.png')
# image = cv2.imread('../../databases/img/cohn/S063_002_00000023.png')
fd = Input(image)
fd.detect_faces()
for face in fd.faces:
    face_img = face.resize((100, 100))
    res = ed.query_no_data(face_img)
    print(res)
    show(face.img)
