import cv2
import numpy as np

class EmotionVisualisation:

    def __init__(self):
        self.row_height = 12
        self.margin = 2
        pass

    def get_img(self, img, params):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        visual = np.zeros((img.shape[0], len(params) * self.row_height, 3), np.uint8)

        #visual[0:img.shape[0], 0:img.shape[1]] = img
        y = 0

        maxval = params[max(params, key=params.get)]
        print(params)
        print('max:')
        print(maxval)
        for param in params:
            #print(param)
            #print(params[param])
            self.draw_bar(visual, y, param, params[param] / maxval)
            y += self.row_height
        #import pdb; pdb.set_trace()

        result = np.concatenate((img, visual), axis=1)
        return result

    def draw_bar(self, img, y, text, value_percent, color=(120, 120, 120)):
        text = text.upper()
        print('valpercent: %s' % int(img.shape[0]))
        cv2.rectangle(img, (0, y + self.margin), (int(img.shape[0] * value_percent), y + self.row_height), color, thickness=cv2.FILLED)
        cv2.putText(img, text, (0, y + self.row_height), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))