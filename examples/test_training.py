from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNConfiguration, FaceImg
from nnemotions.network.nn_functions import SigmoidActivationFunction, QuadraticCostFunction


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)


nn_config = NNConfiguration(bias=True, layersizes=[944, 200, 7],
                            activation_function=SigmoidActivationFunction,
                            cost_function=QuadraticCostFunction,
                            blocksize=25,
                            learningrate=0.2)


#env.db.add(nn_config)
#env.db.commit()


#nn_config = env.db.query(NNConfiguration).get(1)

images = env.db.query(FaceImg).all()#[:10]

trainingHelper = NNTrainingHelper(env, nn_config)

trainingHelper.new_session()

# what to do for do while?
lastcost = 100
while lastcost > 0.001:
    trainingHelper.train(images)
    lastcost = sum(trainingHelper.costs) / len(trainingHelper.costs)
    print(lastcost)

print(trainingHelper.test(images[2:3]))
#trainingHelper.save_network('testing 1 iteration jaffe')