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

print('[{}] Score: {}'.format(nn_training.id, nn_training.score))
fig, ax = plt.subplots()
ax.set_ylabel('Kosten')
ax.set_xlabel('Trainingsbeispiel')
plt.plot(nn_training.train_costs)
plt.plot(nn_training.test_costs)
plt.show()
