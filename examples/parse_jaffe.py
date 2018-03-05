from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
import cv2
from nnemotions.detection.face.input import Input

# script to save faces from the jaffe database:
# http://www.kasrl.org/jaffedb_info.html

# directory the original images are in
DB_ORG_IMG_DIR = '../../databases/jaffe/'
# database to store face info and emotions
NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
# directory to save scaled images in
NN_EMOT_IMG_DIR = '../../databases/img/'
# pixels faces are scaled to
IMG_SIZE = (100, 100)

# open database and start session
engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
Base.metadata.drop_all()
Base.metadata.create_all()
DBSession = sessionmaker(bind=engine)
session = DBSession()



# emotions used in jaffe database
emotions = {
    'HA': 'happy',
    'SA': 'sadness',
    'SU': 'surprised',
    'AN': 'angry',
    'DI': 'disgusted',
    'FE': 'fear'
}

# Write available emotions to table for better linking
for emot_short in emotions:
    session.add(Emotion(tag=emot_short, name=emotions[emot_short], db_name='jaffe'))
session.commit()

# scan directory for images, detect faces, scale faces, store their information in the db
for file in os.listdir(DB_ORG_IMG_DIR):
    if file.endswith('.tiff'):
        for emot in session.query(Emotion).all():
            if emot.tag in file:
                print('adding image ' + file + ' : ' + emot.tag)

                img = cv2.imread(os.path.join(DB_ORG_IMG_DIR, file))
                face_detection = Input(img)
                face_detection.detect_faces()
                if len(face_detection.faces) == 0:
                    print('No faces found in ' + file + ' : ' + emot.tag)
                else:
                    # save scaled image to directory
                    face = face_detection.faces[0].resize(IMG_SIZE)
                    cv2.imwrite(os.path.join(NN_EMOT_IMG_DIR, file), face)
                    # store info in database
                    img = FaceImg(src=file, emotion=emot, db_name='jaffe')
                    session.add(img)

session.commit()
