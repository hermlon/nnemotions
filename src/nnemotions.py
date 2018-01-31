import cv2
from face_detection.input import Input
from face_detection.face import Face

def close():
    cv2.destroyAllWindows()


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    close()


testpath = '../../databases/jaffe/KA.AN1.39.tiff'
testimg = cv2.imread(testpath)
fd = Input(testimg, path=testpath)
fd.detect_faces()
show(fd.faces[0].img)
#import pdb; pdb.set_trace()