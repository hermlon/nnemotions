from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, Float, PickleType, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import cv2
import os

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

    def get_img(self, dir):
        return cv2.imread(os.path.join(dir, self.src), 1)


class NNTraining(Base):
    __tablename__ = 'nntraining'
    id = Column(Integer, primary_key=True)
    learningrate = Column(Float)
    training_iterations = Column(Integer)
    testing_iterations = Column(Integer)
    blocksize = Column(Integer)
    bias = Column(Boolean)
    activation_function = Column(String(80))
    cost_function = Column(String(80))
    layersizes = Column(String(80))
    nn_saved_name = Column(String(120))
    score = Column(Float)
    info = Column(String(120))
    start = Column(DateTime)
    end = Column(DateTime)


engine = create_engine('sqlite:///../../databases/nnemotions.db')

Base.metadata.create_all(engine)
