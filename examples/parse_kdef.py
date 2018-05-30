import uuid
import os
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
import cv2
from nnemotions.detection.face.input import Input
from nnemotions.env import env

# script to save faces from the cohn canade database:
# http://www.consortium.ri.cmu.edu/ckagree/index.cgi

# directory the original images are in
DB_ORG_IMG_DIR = '../../databases/KDEF'

# pixels faces are scaled to
IMG_SIZE = (100, 100)


# emotions used in cohn database and corresponding emotion ids
emotion_tags = {
    'HAS': 1,
    'SAS': 2,
    'SUS': 3,
    'ANS': 4,
    'DIS': 5,
    'AFS': 7
}

for session_dir, other_dirs, files in os.walk(DB_ORG_IMG_DIR):
    for file in files:
        for tag in emotion_tags:
            if tag in file:
                emotion = env.db.query(Emotion).get(emotion_tags[tag])

                print('adding image ' + file + ' : ' + emotion.name)

                img = cv2.imread(os.path.join(NN_EMOT_IMG_DIR, session_dir, file))
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
                    img = FaceImg(src=filename, emotion=emotion, db_name='kdef')
                    env.db.add(img)

env.db.commit()
