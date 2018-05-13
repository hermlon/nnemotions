from nnemotions.util.nn_resenv import NNResEnv
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNConfiguration, FaceImg


NN_EMOT_DB = 'sqlite:///../../databases/nnemotions.db'
NN_EMOT_IMG_DIR = '../../databases/img'
NN_MODEL_DIR = '../../databases/nn_models'

env = NNResEnv(NN_EMOT_DB, NN_MODEL_DIR, NN_EMOT_IMG_DIR)

"""
nn_config = NNConfiguration(bias=True, layersizes=[944, 200, 8],
                            activation_function=SigmoidActivationFunction,
                            cost_function=LinearCostFunction,
                            blocksize=25,
                            learningrate=0.3)


env.db.add(nn_config)
env.db.commit()
"""

nn_config = env.db.query(NNConfiguration).get(1)

images = env.db.query(FaceImg).all()

trainingHelper = NNTrainingHelper(env, nn_config)

trainingHelper.new_session()
trainingHelper.train(images[:len(images)//2])
trainingHelper.test(images[len(images)//2:])
trainingHelper.save_network('testing 1 iteration jaffe')