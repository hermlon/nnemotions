from nnemotions.network.feed_forward_nn import FeedForwardNN
from sqlalchemy.orm import sessionmaker
from nnemotions.detection.emotion.nnemo_db import NNTraining
from nnemotions.detection.emotion.local_binary_pattern import BinaryPatternAnalysis
import numpy
import pickle
import datetime
import uuid
import os


class LBPEmotionDetection:

    def __init__(self, engine, nn_save_path):
        DBSession = sessionmaker(bind=engine)
        self.db_session = DBSession()
        self.nn_save_path = nn_save_path

        self.new_session()

    def save_network(self, info=''):
        file_name = str(uuid.uuid4()) + '.p'
        f = open(os.path.join(self.nn_save_path, file_name), 'wb')
        pickle.dump(self.nn, f)

        nnt = NNTraining(learningrate=self.nn.learningrate,
                         training_iterations=self.training_iterations,
                         testing_iterations=self.testing_iterations,
                         blocksize=self.blocksize,
                         bias=self.nn.bias,
                         activation_function=self.nn.activation_function.name,
                         cost_function=self.nn.cost_function.name,
                         layersizes=str(self.nn.layersizes),
                         nn_saved_name=file_name,
                         score=self.testing_score/self.testing_iterations*100,
                         info=info,
                         start=self.start,
                         end=datetime.datetime.now())
        self.db_session.add(nnt)
        self.db_session.commit()

    def load_network(self, nntraining):
        f = open(os.path.join(self.nn_save_path, nntraining.nn_saved_name), 'rb')
        self.nn = pickle.load(f)
        self.blocksize = nntraining.blocksize
        self.new_session()

    def new_network(self, layersizes, activation_function, cost_function, bias, learningrate, blocksize):
        self.nn = FeedForwardNN(layersizes=layersizes, activation_function=activation_function, bias=bias,
                          cost_function=cost_function, learningrate=learningrate)
        self.blocksize = blocksize
        self.new_session()

    def query(self, img, des_out, learn=False):
        input_nodes = self.local_binary_histogram(img)
        desired_output = numpy.array(des_out, ndmin=2).T
        if learn:
            res = self.nn.train(input_nodes, desired_output)
            self.costsum += self.nn.cost
            self.training_iterations += 1
        else:
            res = self.nn.query(input_nodes)
            self.testing_iterations += 1

        if numpy.argmax(desired_output) == numpy.argmax(res):
            if learn:
                self.training_score += 1
            else:
                self.testing_score += 1

    def query_no_data(self, img):
        input_nodes = self.local_binary_histogram(img)
        res = self.nn.query(input_nodes)
        return res

    def new_session(self):
        self.start = datetime.datetime.now()
        self.costsum = 0
        self.training_iterations = 0
        self.testing_iterations = 0
        self.training_score = 0
        self.testing_score = 0

    def local_binary_histogram(self, img):
        bpa = BinaryPatternAnalysis(img, (self.blocksize, self.blocksize))
        hist = numpy.array(bpa.get_histogram(), ndmin=2).T
        return hist / hist.max()
