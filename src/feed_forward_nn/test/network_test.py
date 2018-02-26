import unittest
import numpy
import random
from feed_forward_nn.network import Network
from feed_forward_nn.activation_functions import LinearFunction, SigmoidFunction
from feed_forward_nn.cost_functions import CostFunctions


class NetworkTestCase(unittest.TestCase):
    """Tests for 'network.py'."""

    def test_feed_forward(self):
        """Are values passed correctly trough the network?"""
        testinput = numpy.array([0.6, 0.8], ndmin=2).T

        testweights_layer1 = numpy.array([[0.4, 0.1],
                                          [0.5, 0.2],
                                          [0.6, 0.3]])
        testweights_layer2 = numpy.array([[0.2, 0.45, 0.3],
                                          [0.1, 0.7, 0.22]])

        testnodes_layer1 = numpy.array([0.32, 0.46, 0.6], ndmin=2).T
        testnodes_layer2 = numpy.array([0.451, 0.486], ndmin=2).T

        nn = Network(layersizes=[2, 3, 2], activation_function=LinearFunction)
        nn.layers[1].weights = testweights_layer1
        nn.layers[2].weights = testweights_layer2
        nn.query(testinput)

        numpy.testing.assert_array_almost_equal(nn.layers[1].nodes, testnodes_layer1)
        numpy.testing.assert_array_almost_equal(nn.layers[2].nodes, testnodes_layer2)

    def test_propagate_backward(self):
        """Are errors passed correctly backward trough the network?"""
        testinput = numpy.array([0.6, 0.8], ndmin=2).T
        testresults = numpy.array([0.1, 0.9], ndmin=2).T

        testweights_layer1 = numpy.array([[0.4, 0.1],
                                          [0.5, 0.2],
                                          [0.6, 0.3]])

        testweights_layer2 = numpy.array([[0.2, 0.45, 0.3],
                                          [0.1, 0.7, 0.22]])

        testerrors_layer1 = numpy.array([0.041779, 0.17541764, 0.074667], ndmin=2).T
        testerrors_layer2 = numpy.array([0.123201, 0.171396], ndmin=2).T

        nn = Network(layersizes=[2, 3, 2], activation_function=LinearFunction)
        nn.layers[1].weights = testweights_layer1
        nn.layers[2].weights = testweights_layer2
        # learning rate 0 means weights won't be adjusted
        nn.train(testinput, testresults, learninrate=0)

        numpy.testing.assert_array_almost_equal(nn.layers[1].errors, testerrors_layer1)
        numpy.testing.assert_array_almost_equal(nn.layers[2].errors, testerrors_layer2)

    def test_weight_adjustment(self):
        """Are the weights adjusted correctly?"""

        testinput = numpy.array([0.6, 0.8], ndmin=2).T
        testresults = numpy.array([0.1, 0.9], ndmin=2).T

        testweights_layer1 = numpy.array([[0.4, 0.1],
                                          [0.5, 0.2],
                                          [0.6, 0.3]])

        testweights_layer2 = numpy.array([[0.2, 0.45, 0.3],
                                          [0.1, 0.7, 0.22]])

        # testnodes_layer1 = numpy.array([0.32, 0.46, 0.6], ndmin=2).T
        # testnodes_layer2 = numpy.array([0.451, 0.486], ndmin=2).T

        # testerrors_layer1 = numpy.array([0.041779, 0.17541764, 0.074667], ndmin=2).T
        # testerrors_layer2 = numpy.array([0.123201, 0.171396], ndmin=2).T

        # 1: derivative of Linear function
        # deltaww = a_L-1 * error
        test_adjusted_weights_layer1 = numpy.array([[0.4 - 0.02506739, 0.1 - 0.0334232],
                                                     [0.5 - 0.1052505, 0.2 - 0.140334112],
                                                     [0.6 - 0.0448002, 0.3 - 0.0597336]])

        test_adjusted_weights_layer2 = numpy.array([[0.2 - 0.03942432, 0.45 - 0.05667246, 0.3 - 0.0739206],
                                                     [0.1 - 0.05484672, 0.7 - 0.07884216, 0.22 - 0.10283759]])

        nn = Network(layersizes=[2, 3, 2], activation_function=LinearFunction)
        nn.layers[1].weights = testweights_layer1
        nn.layers[2].weights = testweights_layer2

        nn.train(testinput, testresults, learninrate=1)

        numpy.testing.assert_array_almost_equal(nn.layers[1].weights, test_adjusted_weights_layer1)
        numpy.testing.assert_array_almost_equal(nn.layers[2].weights, test_adjusted_weights_layer2)

    def test_training_xor(self):
        """Could I have managed to write an nn which is capable of learning xor? ....nope"""

        nn = Network(layersizes=[2, 2, 1], activation_function=SigmoidFunction, bias=True)

        def xor(a, b):
            if a == b:
                return -1
            else:
                return 1

        valuespos = [-1, 1]

        for round in range(100):
            randinp = [valuespos[random.getrandbits(1)], valuespos[random.getrandbits(1)]]
            nn.train(numpy.array(randinp, ndmin=2).T, xor(randinp[0], randinp[1]), learninrate=0.3, cost_function=CostFunctions.linear)

        """
        print(nn.query(numpy.array([1, -1], ndmin=2).T))
        print(nn.query(numpy.array([-1, 1], ndmin=2).T))
        print(nn.query(numpy.array([1, 1], ndmin=2).T))
        print(nn.query(numpy.array([-1, -1], ndmin=2).T))"""


if __name__ == '__main__':
    unittest.main()
