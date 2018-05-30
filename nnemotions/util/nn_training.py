from nnemotions.util.lbp_analysis import LBPAnalysis
from nnemotions.detection.emotion.nnemo_db import NNTraining, Emotion
import pickle
import datetime
import uuid
import os
import random
import cv2
import numpy


class NNTrainingHelper:

    def __init__(self, nn_resenv, nn_config):
        self.env = nn_resenv
        self.nn_config = nn_config
        self.lbpa = LBPAnalysis(self.nn_config)
        self.new_session()
        self.info = False

    def train(self, face_imgs):
        results = []
        # train in random order
        random.shuffle(face_imgs)
        for face_img in face_imgs:
            desired_output = self.generate_desired_outputs(face_img)

            result = self.lbpa.query(self.read_image(face_img), desired_output)

            self.costs.append(self.lbpa.nn.cost)
            # normalize maximum to 1
            result = result / numpy.max(result)
            # convert 2dim array to plain list
            results.append(list(numpy.hstack(result)))
            self.training_iterations += 1
            if numpy.argmax(desired_output) == numpy.argmax(result):
                self.training_score += 1

            self.print_status('Training', self.training_iterations, self.training_score, len(face_imgs))

        return results

    def test(self, face_imgs):
        results = []
        for face_img in face_imgs:
            desired_output = self.generate_desired_outputs(face_img)
            result = self.lbpa.query(self.read_image(face_img))
            # normalize maximum to 1
            result = result / numpy.max(result)
            # convert 2dim array to plain list
            results.append(list(numpy.hstack(result)))
            self.testing_iterations += 1
            if numpy.argmax(desired_output) == numpy.argmax(result):
                self.testing_score += 1

            self.print_status('Testing', self.testing_iterations, self.testing_score, len(face_imgs))

        return results

    def new_session(self):
        self.start = datetime.datetime.now()
        self.training_iterations = 0
        self.testing_iterations = 0
        self.training_score = 0
        self.testing_score = 0
        self.costs = []

    def read_image(self, face_img):
        # read greyscale image from env img dir at specific source
        return cv2.imread(os.path.join(self.env.img_dir, face_img.src), 0)

    def save_network(self, info='', minscore=0.0):
        print(self.costs)
        score = self.testing_score / self.testing_iterations * 100
        nn_saved_name = 'deleted'
        if score > minscore:
            file_name = str(uuid.uuid4()) + '.p'
            f = open(os.path.join(self.env.model_dir, file_name), 'wb')
            pickle.dump(self.lbpa.nn, f)
            nn_saved_name = file_name

        nnt = NNTraining(training_iterations=self.training_iterations,
                         testing_iterations=self.testing_iterations,
                         nn_saved_name=nn_saved_name,
                         score=round(score, 2),
                         info=info,
                         costs=self.costs,
                         start=self.start,
                         end=datetime.datetime.now(),
                         configuration=self.nn_config)
        self.env.db.add(nnt)
        self.env.db.commit()

    # generate array of desired outputs. maybe somehow out of place?!
    def generate_desired_outputs(self, face_img):
        emotions = self.env.db.query(Emotion).all()
        des_o = []
        for emotion in emotions:
            if face_img.emotion == emotion:
                des_o.append(1)
            else:
                des_o.append(0)

        return numpy.array(des_o, ndmin=2).T

    # overwriting the nn object and hoping it has a compatible configuration
    def load_network(self, nntraining):
        f = open(os.path.join(self.env.model_dir, nntraining.nn_saved_name), 'rb')
        self.lbpa.nn = pickle.load(f)

    def print_status(self, mode, current, score, total):
        if self.info:
            print('{}: {:.1%} Score: {:.1%} Cost: {}'.format(mode, current / total, score / total, self.costs[-1]))