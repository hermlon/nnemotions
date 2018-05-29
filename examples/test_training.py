from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNConfiguration, FaceImg
from nnemotions.network.nn_functions import SigmoidActivationFunction, QuadraticCostFunction

import matplotlib.pyplot as plt


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)

"""
nn_config = NNConfiguration(bias=True, layersizes=[944, 200, 7],
                            activation_function=SigmoidActivationFunction,
                            cost_function=QuadraticCostFunction,
                            blocksize=25,
                            learningrate=0.2)

"""
#env.db.add(nn_config)
#env.db.commit()


nn_config = env.db.query(NNConfiguration).get(68)

img_train = env.db.query(FaceImg).filter_by(db_name='cohn').all()
img_test = env.db.query(FaceImg).all()

trainingHelper = NNTrainingHelper(env, nn_config)

trainingHelper.new_session()

# what to do for do while?
costs = []
#while lastcost > 0.9:
try:
    for i in range(1):
        trainingHelper.train(img_train)
        costs += trainingHelper.costs# / len(trainingHelper.costs)
        print(i)
except(KeyboardInterrupt):
    print('Cancelling Training')

plt.plot(costs)
plt.show()
#print(trainingHelper.test(img_test))
#trainingHelper.save_network('min cost: 0.9; train cohn; testing all')