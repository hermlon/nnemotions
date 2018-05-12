from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNConfiguration, Emotion
from nnemotions.network.nn_functions import SigmoidActivationFunction, LinearCostFunction

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../../databases/img/'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)

nn_config = NNConfiguration(bias=True, layersizes=[8, 5, 2],
                            activation_function=SigmoidActivationFunction,
                            cost_function=LinearCostFunction,
                            blocksize=25,
                            learningrate=0.3)


env.db.add(nn_config)
env.db.commit()

trainingHelper = NNTrainingHelper(env, nn_config)