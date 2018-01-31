import cv2

class Face:

    def __init__(self, inputimg, startx, starty, endx, endy, resize, path=None, nr=None):
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy
        self.img = inputimg[self.starty:self.endy, self.startx:self.endx]

        if resize is not None:
            self.img = cv2.resize(self.img, (resize, resize), interpolation=cv2.INTER_AREA)

        if path is not None:
            self.save(path + str(nr) + 'scaled.png')

    def save(self, path):
        print('saving to ' + path)
        cv2.imwrite(path, self.img)
