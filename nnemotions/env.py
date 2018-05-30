from nnemotions.util.nn_resenv import NNResEnv

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img/'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)