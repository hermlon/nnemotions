from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from databases.nnemo_db import Base, NNTraining
from emotion_detection.lbp_emotion_detection import LBPEmotionDetection
from picamera.array import PiRGBArray
from picamera import PiCamera
from face_detection.input import Input
import time
import cv2


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

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# run through face detection
face_detection = Input(image)
face_detection.detect_faces()
face = face_detection.faces[0].resize((100, 100))


res = ed.query_no_data(face)
print(res)
show(face)