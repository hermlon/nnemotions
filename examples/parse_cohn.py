import uuid
import os
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
import cv2
from nnemotions.detection.face.input import Input
from nnemotions.env import env

# script to save faces from the cohn canade database:
# http://www.consortium.ri.cmu.edu/ckagree/index.cgi

# directory the original images are in
DB_ORG_IMG_DIR = '../../databases/cohn_kanade/cohn-kanade-images'
# directory the emotions are in
DB_ORG_EMO_DIR = '../../databases/cohn_kanade/Emotion'
# pixels faces are scaled to
IMG_SIZE = (100, 100)


# emotions used in cohn database and corresponding emotion ids
emotion_tags = {
    '5': 1,
    '6': 2,
    '7': 3,
    '1': 4,
    '3': 5,
    '4': 6,
    '2': 7
}

for emotion_dir, session_dirs, f1 in os.walk(DB_ORG_EMO_DIR):
    for session_dir_name in session_dirs:
        for session_dir, record_dirs, f2 in os.walk(os.path.join(emotion_dir, session_dir_name)):
            for record_dir_name in record_dirs:
                for record_dir, d1, record_file in os.walk(os.path.join(session_dir, record_dir_name)):
                    if len(record_file) == 1:
                        split_name = record_file[0].split('_')
                        em_session = split_name[0]
                        em_record = split_name[1]
                        img_file_name = record_file[0].split('_emotion.txt')[0] + '.png'

                        img_path = os.path.join(DB_ORG_IMG_DIR, em_session, em_record, img_file_name)

                        # read only the first 4 characters, because the 4th is the one we look for
                        filecontent = open(os.path.join(record_dir, record_file[0]), 'r').read(4)
                        for tag in emotion_tags:
                            if tag in filecontent:
                                emotion = env.db.query(Emotion).get(emotion_tags[tag])

                                print('adding image ' + img_file_name + ' : ' + emotion.name)

                                img = cv2.imread(img_path)
                                face_detection = Input(img)
                                face_detection.detect_faces()
                                if len(face_detection.faces) == 0:
                                    print('No faces found in ' + img_file_name + ' : ' + emotion.name)
                                else:
                                    # save scaled image to directory
                                    face = face_detection.faces[0].get_image(size=IMG_SIZE, grayscale=True)
                                    # write image to image folder, name is a uuid
                                    filename = str(uuid.uuid4()) + '.png'
                                    cv2.imwrite(os.path.join(env.img_dir, filename), face)
                                    print(os.path.join(env.img_dir, filename))
                                    # store info in database
                                    img = FaceImg(src=filename, emotion=emotion, db_name='cohn')
                                    env.db.add(img)

env.db.commit()
