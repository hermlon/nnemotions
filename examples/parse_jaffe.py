import uuid
import os
from nnemotions.detection.emotion.nnemo_db import FaceImg, Emotion
import cv2
from nnemotions.detection.face.input import Input
from nnemotions.util.nn_resenv import NNResEnv

# script to save faces from the jaffe database:
# http://www.kasrl.org/jaffedb_info.html

# directory the original images are in
DB_ORG_IMG_DIR = '../../databases/jaffe'
# pixels faces are scaled to
IMG_SIZE = (100, 100)

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)


# emotions used in jaffe
# database and the corresponding ids
emotion_tags = {
    'HA': 1,
    'SA': 2,
    'SU': 3,
    'AN': 4,
    'DI': 5,
    'FE': 7
}

# scan directory for images, detect faces, scale faces, store their information in the db

for file in os.listdir(DB_ORG_IMG_DIR):
    if file.endswith('.tiff'):
        for tag in emotion_tags:
            if tag in file:
                emotion = env.db.query(Emotion).get(emotion_tags[tag])

                print('adding image ' + file + ' : ' + emotion.name)

                img = cv2.imread(os.path.join(DB_ORG_IMG_DIR, file))
                face_detection = Input(img)
                face_detection.detect_faces()
                if len(face_detection.faces) == 0:
                    print('No faces found in ' + file + ' : ' + emotion.name)
                else:
                    # save scaled image to directory
                    face = face_detection.faces[0].get_image(size=IMG_SIZE, grayscale=True)
                    # write image to image folder, name is a uuid
                    filename = str(uuid.uuid4()) + '.png'
                    cv2.imwrite(os.path.join(env.img_dir, filename), face)
                    print(os.path.join(env.img_dir, filename))
                    # store info in database
                    img = FaceImg(src=filename, emotion=emotion, db_name='jaffe')
                    env.db.add(img)

env.db.commit()
