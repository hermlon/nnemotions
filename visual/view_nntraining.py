import matplotlib.pyplot as plt
from nnemotions.detection.emotion.nnemo_db import NNTraining
from nnemotions.util.nn_resenv import NNResEnv
from sqlalchemy import desc, asc
from nnemotions.env import env


print('id of nntraining to load:')
nn_training_id = input()
if nn_training_id == '':
    nn_training = env.db.query(NNTraining).order_by(desc('end')).first()
else:
    nn_training = env.db.query(NNTraining).get(nn_training_id)

plt.plot(nn_training.costs)
plt.show()
