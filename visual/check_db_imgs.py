from nnemotions.env import env
import cv2
import os
from nnemotions.detection.emotion.nnemo_db import FaceImg

def show(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgs = env.db.query(FaceImg).filter_by(db_name='kdef')[20:40]

for img in imgs:
    image = cv2.imread(os.path.join(env.img_dir, img.src), 0)
    show(image, img.emotion.name)
