from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from nnemotions.network.nn_functions import SigmoidFunction, ReLuFunction
from nnemotions.network.cost_functions import Linear, Quadratic
import random


def query(tries, learn, data):
    for i in range(tries):
        training_example = data[random.randint(0, len(data)-1)]

        des_o = []
        for emotion in emotions:
            if training_example.emotion == emotion:
                des_o.append(1)
            else:
                des_o.append(0)

        ed.query(training_example.get_img(NN_EMOT_IMG_DIR), des_o, learn=learn)
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

emotions = session.query(Emotion).filter_by(db_name='cohn').all()

cohn_imgages = session.query(FaceImg).filter_by(db_name='cohn').order_by(FaceImg.id).all()

ed = LBPEmotionDetection(engine, NN_MODEL_DIR)

def try_configuration(layersizes, activation_function, cost_function, bias, blocksize, learningrate, training_cycles):
    ed.new_network(layersizes=layersizes, activation_function=activation_function, cost_function=cost_function, bias=bias,
                   blocksize=blocksize, learningrate=learningrate)
    ed.new_session()
    print('Training...')
    query(training_cycles, True, cohn_imgages[:len(cohn_imgages)//2])
    print('Testing...')
    query(500, False, cohn_imgages[len(cohn_imgages)//2:])

    ed.save_network('training: cohn, all emotions', minscore=60.0)


# (25,25) 944 = 59 * 4 * 4
# (5,5) 23600 = 59 * 20 * 20
cost_function = Linear
for learningrate in [x / 100.0 for x in range(1, 26, 2)]:
    for activation_func in [SigmoidFunction, ReLuFunction]:
        try_configuration([944, 300, 200, 150, 80, len(emotions)], activation_func, cost_function, True, 25, learningrate, 900)
        try_configuration([944, 500, 80, len(emotions)], activation_func, cost_function, True, 25, learningrate, 600)
        try_configuration([23600, 1000, 300, 40, len(emotions)], activation_func, cost_function, True, 5, learningrate, 700)

        try_configuration([944, 700, 600, 300, 200, 80, 35, len(emotions)], activation_func, cost_function, True, 25,
                          learningrate, 900)
        try_configuration([944, 150, 30, len(emotions)], activation_func, cost_function, True, 25, learningrate, 600)
        try_configuration([23600, 2000, 1500, 1000, 450, 150, 70, len(emotions)], activation_func, cost_function, True, 5, learningrate,
                          700)

# try_configuration([944, 300, 80, 20, len(emotions)], SigmoidFunction, Linear, True, 25, 0.21, 15000)
