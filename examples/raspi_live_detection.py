from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, NNTraining
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from picamera.array import PiRGBArray
from picamera import PiCamera
from nnemotions.detection.face.input import Input
import time
import cv2


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/'
NN_MODEL_DIR = '../../databases/nn_models'


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_results(results, face, img):
	height = 10
	y = face.endy
	for result in results:
		width = face.endx - face.startx
		cv2.rectangle(img, (face.startx, y + int(height*1/3)), (face.startx + int(result * width), y + int(height*2/3)), (255,0,0), int(height*1/3))
		y += height


engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

nntraining = session.query(NNTraining).first()
print(nntraining.info)
ed = LBPEmotionDetection(engine, NN_MODEL_DIR)
ed.load_network(nntraining)


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	
	# run through face detection
	face_detection = Input(image)
	face_detection.detect_faces()
	for face in face_detection.faces:
		cv2.rectangle(image,(face.startx, face.starty),(face.endx, face.endy), (255,0,0), 2)
		faceimg = face.resize((100, 100))

		res = ed.query_no_data(faceimg)
		print_results(res, face, image)
		print(res)

	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
