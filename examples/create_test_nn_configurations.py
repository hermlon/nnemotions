from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.detection.emotion.nnemo_db import NNConfiguration
from nnemotions.network.nn_functions import *

NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../../databases/img/'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)

# Generate some configurations for testing

blocksize = 25
bias = True

for activation_function in [SigmoidActivationFunction, ReLuActivationFunction]:
    for cost_function in [LinearCostFunction, QuadraticCostFunction]:
        for learningrate in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 3.0]:
            for layersizes in [(944, 300, 200, 150, 80, 7), (944, 500, 80, 7), (944, 700, 600, 300, 200, 80, 35, 7), (944, 150, 30, 7)]:
                env.db.add(NNConfiguration(blocksize=blocksize, bias=bias, activation_function=activation_function, cost_function=cost_function, learningrate=learningrate, layersizes=layersizes))

env.db.commit()