import cv2
import numpy as np

class EmotionVisualisation:

    def __init__(self):
        self.row_height = 12
        self.margin = 2
        pass

    def get_img(self, img, params):
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        visual = np.zeros((len(params) * self.row_height, img.shape[0], 3), np.uint8)

        #visual[0:img.shape[0], 0:img.shape[1]] = img
        y = 0

        maxval = params[max(params, key=params.get)]
        color_i = 0
        colors = [(41, 205, 255), (245, 73, 46), (208, 52, 219), (17, 27, 166), (46, 107, 21), (119, 16, 135), (246, 246, 126)]
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