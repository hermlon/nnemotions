from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from nnemotions.network.activation_functions import SigmoidFunction, ReLuFunction
from nnemotions.network.cost_functions import Linear, Quadratic
import random


def query(tries, learn, happy, sad):
    for i in range(tries):
        # 1 means we train a happy one!
        if bool(random.getrandbits(1)):
            des_o = [1, 0]
            img = happy[random.randint(0, len(happy)-1)]
        else:
            des_o = [0, 1]
            img = sad[random.randint(0, len(sad)-1)]

        ed.query(img.get_img(NN_EMOT_IMG_DIR), des_o, learn=learn)
        if i % 50 == 0 or i == tries:
            percent = i / tries * 100
            print('> %.2f%% TrainS: %d%% TestS: %d%%' % (percent, ed.training_score/(ed.training_iterations+0.001)*100, ed.testing_score/(ed.testing_iterations+0.001)*100))


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/cohn'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


cohn_imgages = session.query(FaceImg).filter_by(db_name='cohn')
happy_images = cohn_imgages.filter(FaceImg.emotion.has(tag='5')).order_by(FaceImg.id.desc()).all()
sad_images = cohn_imgages.filter(FaceImg.emotion.has(tag='6')).order_by(FaceImg.id.desc()).all()


training_data_happy = happy_images[:61]
training_data_sad = sad_images[:20]
test_data_happy = happy_images[61:69]
test_data_sad = sad_images[20:28]

ed = LBPEmotionDetection(engine)

training_cycles = 600

ed.new_network(layersizes=[944, 100, 30, 2], activation_function=SigmoidFunction, cost_function=Linear, bias=True, blocksize=25, learningrate=0.3)
ed.new_session()
print('Training...')
query(training_cycles, True, training_data_happy, training_data_sad)
print('Testing...')
query(500, False, test_data_happy, test_data_sad)
ed.save_network('training various parameters with cohn dataset')
