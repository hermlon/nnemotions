import cv2
from face_detection.face import Face


class Input:

    def __init__(self, img, path=None):
        self.faces = list()
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.path = path

    def detect_faces(self, resize=50):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        i = 0
        for (x, y, w, h) in detected_faces:
            self.faces.append(Face(self.gray, x, y, x + w, y + h, resize, path=self.path, nr=i))
            i += 1
