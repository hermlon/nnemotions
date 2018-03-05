import cv2

class Face:

    def __init__(self, inputimg, startx, starty, endx, endy):
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy
        self.img = inputimg[self.starty:self.endy, self.startx:self.endx]

    def resize(self, size):
        self.img = cv2.resize(self.img, size, interpolation=cv2.INTER_AREA)
        return self.img
