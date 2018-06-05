import cv2
from nnemotions.detection.face.input import Input
from nnemotions.emotion_visualisation import EmotionVisualisation
from nnemotions.env import env
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNTraining, Emotion


def close():
    cv2.destroyAllWindows()


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    close()


testpath = '../../databases/test_img/f8.jpeg'
testimg = cv2.imread(testpath, 1)
fd = Input(testimg)
fd.detect_faces()

nntraining = env.db.query(NNTraining).get(39)

th = NNTrainingHelper(env, nntraining.configuration)
th.load_network(nntraining)

emotions = env.db.query(Emotion).all()

for face in fd.faces[:1]:
    data = {}
    img = face.get_image(size=(100,100), grayscale=True)
    res = th.query(img)
    print(res)

    i = 0
    for e in emotions:
        data[e.name] = res[i]
        i += 1
    info_img = EmotionVisualisation().get_img(face.img, data)
    show(info_img)
