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

        # workaround for now...
        if learn:
            NN_EMOT_IMG_DIR = '../../databases/img/cohn'
        else:
            NN_EMOT_IMG_DIR = '../../databases/img/jaffe'

        ed.query(img.get_img(NN_EMOT_IMG_DIR), des_o, learn=learn)
        if i % 50 == 0 or i == tries:
            percent = i / tries * 100
            print('> %.2f%% TrainS: %d%% TestS: %d%%' % (percent, ed.training_score/(ed.training_iterations+0.001)*100, ed.testing_score/(ed.testing_iterations+0.001)*100))


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/cohn'
NN_MODEL_DIR = '../../databases/nn_models'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


cohn_imgages = session.query(FaceImg).filter_by(db_name='cohn')
training_data_happy = cohn_imgages.filter(FaceImg.emotion.has(tag='5')).order_by(FaceImg.id.desc()).all()
training_data_sad = cohn_imgages.filter(FaceImg.emotion.has(tag='6')).order_by(FaceImg.id.desc()).all()

jaffe_images = session.query(FaceImg).filter_by(db_name='jaffe')
test_data_happy = jaffe_images.filter(FaceImg.emotion.has(tag='HA')).order_by(FaceImg.id.desc()).all()
test_data_sad = jaffe_images.filter(FaceImg.emotion.has(tag='SA')).order_by(FaceImg.id.desc()).all()


ed = LBPEmotionDetection(engine, NN_MODEL_DIR)

def try_configuration(layersizes, activation_function, cost_function, bias, blocksize, learningrate, training_cycles):
    ed.new_network(layersizes=layersizes, activation_function=activation_function, cost_function=cost_function, bias=bias,
                   blocksize=blocksize, learningrate=learningrate)
    ed.new_session()
    print('Training...')
    query(training_cycles, True, training_data_happy, training_data_sad)
    print('Testing...')
    query(300, False, test_data_happy, test_data_sad)
    ed.save_network('training: cohn, testing: jaffe, learningrate vs activation vs cost vs blocksize')


# (25,25) 944 = 59 * 4 * 4
# (5,5) 23600 = 59 * 20 * 20

for learningrate in [x / 100.0 for x in range(1, 30, 5)]:
    for activation_func in [SigmoidFunction, ReLuFunction]:
        for cost_function in [Linear, Quadratic]:
            try_configuration([944, 300, 80, 20, 2], activation_func, cost_function, True, 25, learningrate, 300)
            try_configuration([23600, 1000, 300, 40, 2], activation_func, cost_function, True, 5, learningrate, 500)
