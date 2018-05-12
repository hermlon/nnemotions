from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.detection.emotion.nnemo_db import Emotion

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../../databases/img/'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)

emotions = ['fröhlich', 'traurig', 'überrascht', 'zornig', 'angewidert', 'verachtend', 'ängstlich']

for name in emotions:
    env.db.add(Emotion(name=name))

env.db.commit()