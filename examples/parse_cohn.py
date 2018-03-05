from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
import cv2
from nnemotions.detection.face.input import Input

# script to save faces from the cohn canade database:
# http://www.consortium.ri.cmu.edu/ckagree/index.cgi

# directory the original images are in
DB_ORG_IMG_DIR = '../../databases/cohn_kanade/cohn-kanade-images'
# directory the emotions are in
DB_ORG_EMO_DIR = '../../databases/cohn_kanade/Emotion'
# database to store face info and emotions
NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
# directory to save scaled images in
NN_EMOT_IMG_DIR = '../../databases/img/cohn'
# pixels faces are scaled to
IMG_SIZE = (100, 100)

# open database and start session
engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
Base.metadata.drop_all()
Base.metadata.create_all()
DBSession = sessionmaker(bind=engine)
session = DBSession()



# emotions used in cohn database
emotions = {
    '5': 'happy',
    '6': 'sadness',
    '7': 'surprised',
    '1': 'angry',
    '3': 'disgusted',
    '4': 'fear',
    '0': 'neutral',
    '2': 'contempt'
}

# Write available emotions to table for better linking
for emot_short in emotions:
    session.add(Emotion(tag=emot_short, name=emotions[emot_short], db_name='cohn'))
session.commit()

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
                        for emot in session.query(Emotion).filter_by(db_name='cohn').all():
                            if emot.tag in filecontent:
                                print(emot.name)
                                print(img_path)
                                print('adding image ' + img_file_name + ' : ' + emot.name)

                                img = cv2.imread(img_path)
                                face_detection = Input(img)
                                face_detection.detect_faces()
                                if len(face_detection.faces) == 0:
                                    print('No faces found!')
                                else:
                                    # save scaled image to directory
                                    face = face_detection.faces[0].resize(IMG_SIZE)
                                    cv2.imwrite(os.path.join(NN_EMOT_IMG_DIR, img_file_name), face)
                                    # store info in database
                                    img = FaceImg(src=img_file_name, emotion=emot, db_name='cohn')
                                    session.add(img)

session.commit()
