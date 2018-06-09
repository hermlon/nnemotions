from nnemotions.env import env
from nnemotions.util.nn_training import NNTrainingHelper
from nnemotions.detection.emotion.nnemo_db import NNConfiguration, FaceImg, Emotion
from nnemotions.network.nn_functions import SigmoidActivationFunction, QuadraticCostFunction

"""
nn_config = NNConfiguration(bias=True, layersizes=[944, 200, 7],
                            activation_function=SigmoidActivationFunction,
                            cost_function=QuadraticCostFunction,
                            blocksize=25,
                            learningrate=0.2)

"""

nn_configs = [env.db.query(NNConfiguration).get(35)]

emotions = [env.db.query(Emotion).get(1), env.db.query(Emotion).get(2)]

img_em = []
for em in emotions:
    img_em.append(env.db.query(FaceImg).filter_by(emotion=em, db_name='jaffe').all())

img_train = []
img_test = []
#max_samples = [150, 150, 150, 150, 150, 20, 150]
max_samples = [20, 20, 20, 20, 20, 0, 20]
# add maximal available images to training data and the rest to testing data
for i in range(len(img_em)):
    img_train += img_em[i][:max_samples[i]]
    img_test += img_em[i][max_samples[i]:]

#img_train = img_train[:5]
#img_test = img_test[:5]

for nn_config in nn_configs:
    print(repr(nn_config))
    trainingHelper = NNTrainingHelper(env, nn_config)
    trainingHelper.new_session()
    try:
        for i in range(10):
            trainingHelper.train(img_train, emotions)
            print('epoch {}'.format(i))
    except(KeyboardInterrupt):
        print('Cancelling Training')
    trainingHelper.test(img_test, emotions)
    trainingHelper.save_network('happy/sad training 20 of jaffe, rest jaffe for testing', minscore=50)
