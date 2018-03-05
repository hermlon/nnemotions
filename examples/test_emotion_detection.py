from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import Base, FaceImg, Emotion
from nnemotions.detection.emotion.lbp_emotion_detection import LBPEmotionDetection
from nnemotions.network.activation_functions import SigmoidFunction, ReLuFunction
from nnemotions.network.cost_functions import Linear, Quadratic


def query(tries, learn, data):
    it = 0
    maxit = tries * len(data)
    for i in range(tries):
        for img in data:
            if img.emotion.tag == 'HA':
                des_o = [1, 0]
            else:
                des_o = [0, 1]

            ed.query(img.get_img(NN_EMOT_IMG_DIR), des_o, learn=learn)
            it += 1

            if it % 50 == 0 or it == maxit:
                percent = it / maxit * 100
                print('> %.2f%% TrainS: %d%% TestS: %d%%' % (percent, ed.training_score/(ed.training_iterations+0.001)*100, ed.testing_score/(ed.testing_iterations+0.001)*100))


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/'

engine = create_engine(NN_EMOT_DB)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


happy = session.query(FaceImg).filter(FaceImg.emotion.has(tag='HA')).order_by(FaceImg.id.desc()).all()
sad = session.query(FaceImg).filter(FaceImg.emotion.has(tag='SA')).order_by(FaceImg.id.desc()).all()

training_data = happy[:20] + sad[:20]
test_data = happy[21:31] + sad[21:31]

ed = LBPEmotionDetection(engine)



training_cycles = 80


ed.new_network(layersizes=[944, 100, 30, 2], activation_function=SigmoidFunction, cost_function=Linear, bias=True, blocksize=25, learningrate=0.3)
ed.new_session()
query(training_cycles, True, training_data)
query(1, False, test_data)
ed.save_network('training various parameters with jaffe dataset')
