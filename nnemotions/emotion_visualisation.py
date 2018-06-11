import cv2
import numpy as np


class EmotionVisualisation:

    def __init__(self):
        self.row_height = 12
        self.margin = 2
        pass

    def get_img(self, img, params):
        visual = np.zeros((len(params) * self.row_height, img.shape[0], 3), np.uint8)
        y = 0

        maxval = params[max(params, key=params.get)]
        color_i = 0
        colors = [(255, 205, 41), (46, 73, 246), (219, 52, 208), (166, 27, 17), (21, 107, 46), (135, 16, 119), (26, 146, 146)]
        for param in params:
            # cycle through colors
            self.draw_bar(visual, y, param, params[param] / maxval, color=colors[color_i%len(colors)])
            y += self.row_height
            color_i += 1

        result = np.concatenate((img, visual), axis=0)
        return result

    def draw_bar(self, img, y, text, value_percent, color=(255, 120, 120)):
        text = text.upper()
        cv2.rectangle(img, (0, y + self.margin), (int(img.shape[1] * value_percent), y + self.row_height), color, thickness=cv2.FILLED)
        cv2.putText(img, text, (0, y + self.row_height), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
