from nnemotions.env import env
from nnemotions.detection.emotion.nnemo_db import Emotion


emotions = ['fröhlich', 'traurig', 'überrascht', 'zornig', 'angewidert', 'verachtend', 'ängstlich']

for name in emotions:
    env.db.add(Emotion(name=name))

env.db.commit()