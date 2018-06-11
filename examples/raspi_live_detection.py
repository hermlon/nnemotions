from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from nnemotions.env import env
from nnemotions.detection.emotion.nnemo_db import NNTraining, Emotion
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.face.input import Input
from nnemotions.emotion_visualisation import EmotionVisualisation
import numpy as np
from collections import OrderedDict

nntraining = env.db.query(NNTraining).get(78)
emotions = env.db.query(Emotion).order_by(Emotion.id.asc()).all()


th = NNTrainingHelper(env, nntraining.configuration)
th.load_network(nntraining)

camera = PiCamera()
camera.resolution = (800, 480)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(800, 480))

cv2.namedWindow("nnemotions", cv2.WND_PROP_FULLSCREEN)          
cv2.setWindowProperty("nnemotions", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = np.array(frame.array).copy()

    fd = Input(image)
    fd.detect_faces()
    
    for face in fd.faces:
        data = OrderedDict()
        img = face.get_image(size=(100,100), grayscale=True)
        res = th.query(img)
        i = 0
        for e in emotions:
            data[e.name] = res[i]
            i += 1

        face_img = face.get_image()
        face_img = face_img // 1.3
        info_img = EmotionVisualisation().get_img(face_img, data)

        endx = info_img.shape[1] + face.startx
        endy = info_img.shape[0] + face.starty
        if endx > image.shape[1]:
            endx = image.shape[1]
        if endy > image.shape[0]:
            endy = image.shape[0]
        image[face.starty:endy, face.startx:endx] = info_img[:endy-face.starty, :endx-face.startx]
    
    cv2.imshow('nnemotions', image)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord(' '):
        break
