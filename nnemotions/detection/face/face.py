import cv2


class Face:

    def __init__(self, inputimg, startx, starty, endx, endy):
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy
        self.img = inputimg[self.starty:self.endy, self.startx:self.endx]

    def get_image(self, size=None, grayscale=False):
        img = self.img
        if grayscale:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if size is None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    # Dict of emotions and corresponding result
    def set_results(self):
        pass

    # get image with stats
    def get_stats(self):
        pass

    # Copy the stats to the given image at face position
    def add_stats(self, img):
        pass

