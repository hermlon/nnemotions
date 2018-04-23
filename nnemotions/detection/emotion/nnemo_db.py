from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, Float, PickleType, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json
from nnemotions.network.cost_functions import cost_functions
from nnemotions.network.activation_functions import activation_functions

Base = declarative_base()


class Emotion(Base):
    __tablename__ = 'emotion'
    id = Column(Integer, primary_key=True)
    name = Column(String(250))
    tag = Column(String(250))
    db_name = Column(String(250))


class FaceImg(Base):
    __tablename__ = 'face_img'
    id = Column(Integer, primary_key=True)
    src = Column(String(250))
    db_name = Column(String(250))
    emotion_id = Column(Integer, ForeignKey('emotion.id'))
    emotion = relationship(Emotion)


class NNConfiguration(Base):
    __tablename__ = 'nnconfiguration'
    id = Column(Integer, primary_key=True)
    layersizes = Column(String(80))
    activation_function = Column(String(80))
    cost_function = Column(String(80))
    bias = Column(Boolean)
    blocksize = Column(Integer)
    learningrate = Column(Float)

    def get_layersizes(self):
        return json.loads(self.layersizes)

    def set_layersizes(self, layersizes):
        self.layersizes = json.dumps(layersizes)

    def get_cost_function(self):
        return cost_functions[self.cost_function]

    def get_activation_function(self):
        return activation_functions[self.activation_function]


class NNTraining(Base):
    __tablename__ = 'nntraining'
    id = Column(Integer, primary_key=True)
    training_iterations = Column(Integer)
    testing_iterations = Column(Integer)
    nn_saved_name = Column(String(120))
    score = Column(Float)
    info = Column(String(120))
    start = Column(DateTime)
    end = Column(DateTime)
    configuration_id = Column(Integer, ForeignKey('NNConfiguration.id'))
    configuration = relationship(NNConfiguration)
