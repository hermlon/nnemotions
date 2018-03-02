from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
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


class FaceImg(Base):
    __tablename__ = 'face_img'
    id = Column(Integer, primary_key=True)
    src = Column(String(250))
    db_name = Column(String(250))
    emotion_id = Column(Integer, ForeignKey('emotion.id'))
    emotion = relationship(Emotion)

    def get_img(self, dir):
        return cv2.imread(os.path.join(dir, self.src))



engine = create_engine('sqlite:///../../../databases/nnemotions.db')

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
