import cv2
import numpy as np

class EmotionVisualisation:

    def __init__(self):
        self.row_height = 10
        pass

    def get_img(self, img, params):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        visual = np.zeros((img.shape[1] + len(params) * self.row_height, img.shape[0], 3))

        #visual[0:img.shape[0], 0:img.shape[1]] = img
        y = 0

        maxval = params[max(params, key=params.get)]
        print(params)
        print('max:')
        print(maxval)
        for param in params:
            #print(param)
            #print(params[param])
            self.draw_bar(img, y, param, params[param] / maxval)
            y += self.row_height
        #import pdb; pdb.set_trace()
        return img

    def draw_bar(self, img, y, text, value_percent, color=(100, 255, 30)):
        text = text[:3].upper()
        print('valpercent: %s' % value_percent)
        cv2.rectangle(img, (0, y), (int(img.shape[0] * value_percent), y + self.row_height), color)
        cv2.putText(img, text + str(value_percent), (0, y + self.row_height), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 50, 200))