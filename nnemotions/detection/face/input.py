import cv2
from nnemotions.detection.face.face import Face


class Input:

    def __init__(self, img):
        self.faces = list()
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        face_cascade = cv2.CascadeClassifier('/home/pi/wpa/nnemotions/venv/lib/python3.5/site-packages/cv2/data/' + 'haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        for (x, y, w, h) in detected_faces:
            self.faces.append(Face(self.gray, x, y, x + w, y + h))
