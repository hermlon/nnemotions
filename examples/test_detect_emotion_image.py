from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, NNTraining, Emotion, FaceImg
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from nnemotions.detection.face.input import Input
from nnemotions.emotion_visualisation import EmotionVisualisation
import cv2

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/cohn/'
NN_MODELS_DIR = '../../databases/nn_models'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

nntraining = session.query(NNTraining).filter_by(id=70).first()
print('Score: %s' % nntraining.score)
ed = LBPEmotionDetection(engine, NN_MODELS_DIR)
ed.load_network(nntraining)

# manually adjustable!
emotions = session.query(Emotion).filter_by(db_name='cohn')

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#image = cv2.imread('../../databases/test_img/f1.jpeg')
# image = cv2.imread('../../databases/img/cohn/S063_002_00000023.png')

face_images = session.query(FaceImg).filter_by(db_name='cohn')

for face_img in face_images[20:40]:
    image = face_img.get_img(NN_EMOT_IMG_DIR)
    fd = Input(image)
    fd.detect_faces()

    for face in fd.faces:
        face_img = face.resize((100, 100))
        res = ed.query_no_data(face_img)

        result_dir = {}
        print('---------------------')
        i = 0
        for r in res:
            result_dir[emotions[i].name] = r
            print(emotions[i].name)
            print(r)
            i += 1

        e = EmotionVisualisation()
        visual = e.get_img(face.img, result_dir)
        show(visual)

# TODO: Kontrast vorher hochziehen!