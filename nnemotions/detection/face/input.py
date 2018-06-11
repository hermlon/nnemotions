import cv2
import os
from nnemotions.detection.face.face import Face


class Input:

    def __init__(self, img):
        self.faces = list()
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        detected_faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in detected_faces:
            self.faces.append(Face(self.img, x, y, x + w, y + h))
