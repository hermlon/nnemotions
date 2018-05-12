from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, Float, PickleType, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json
import sqlalchemy.types
from nnemotions.network.nn_functions import nn_functions
from sqlalchemy.types import TypeDecorator
from sqlalchemy import MetaData
from sqlalchemy import create_engine

Base = declarative_base()



class Emotion(Base):
    __tablename__ = 'emotion'
    id = Column(Integer, primary_key=True)
    name = Column(String(250))


class FaceImg(Base):
    __tablename__ = 'face_img'
    id = Column(Integer, primary_key=True)
    src = Column(String(250))
    db_name = Column(String(250))
    emotion_id = Column(Integer, ForeignKey('emotion.id'))
    emotion = relationship(Emotion)


class LayersizesList(TypeDecorator):

    impl = sqlalchemy.types.VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class NNFunction(TypeDecorator):

    impl = sqlalchemy.types.VARCHAR

    def process_bind_param(self, value, dialect):
        return value.name

    def process_result_value(self, value, dialect):
        return nn_functions[value]



class NNConfiguration(Base):
    __tablename__ = 'nnconfiguration'
    id = Column(Integer, primary_key=True)
    layersizes = Column(LayersizesList)
    activation_function = Column(NNFunction)
    cost_function = Column(NNFunction)
    bias = Column(Boolean)
    blocksize = Column(Integer)
    learningrate = Column(Float)


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
    configuration_id = Column(Integer, ForeignKey('nnconfiguration.id'))
    configuration = relationship(NNConfiguration)



# remove all tables and create all afterwards

#NN_EMOT_DB = 'sqlite:////home/hermon/wpa/databases/nnemotions.db'
#engine = create_engine(NN_EMOT_DB)
#Base.metadata.drop_all(engine)
#Base.metadata.create_all(engine)