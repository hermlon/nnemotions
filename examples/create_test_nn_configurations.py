from nnemotions.env import env
from nnemotions.detection.emotion.nnemo_db import NNConfiguration
from nnemotions.network.nn_functions import *

# Generate some configurations for testing

blocksize = 10
bias = True
activation_function = SigmoidActivationFunction
cost_function = QuadraticCostFunction

for learningrate in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 3.0]:
    for layersizes in [(5900, 2000, 500, 100, 30, 7), (5900, 500, 30, 7), (5900, 2000, 7)]:
        env.db.add(NNConfiguration(blocksize=blocksize, bias=bias, activation_function=activation_function,
                                   cost_function=cost_function, learningrate=learningrate, layersizes=layersizes))

env.db.commit()
