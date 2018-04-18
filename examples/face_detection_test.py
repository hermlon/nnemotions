import cv2
import numpy
import math
from nnemotions.detection.face.input import Input


def close():
    cv2.destroyAllWindows()


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    close()


testpath = '../../databases/img/cohn/S138_007_00000011.png'
testimg = cv2.imread(testpath, 1)
fd = Input(testimg)
fd.detect_faces()


img = fd.faces[0].img


def f(x):
    #x1 = int(x * 2 / 2)
    import pdb;
    #pdb.set_trace()
    return numpy.int_(x*2)#int(((x / 255) ** (1 / 1)) * 255)

img2 = f(img)


print(img2)
show(img2)