import unittest
import numpy
from feed_forward_nn.network import Network
from feed_forward_nn.activation_functions import LinearFunction

class NetworkTestCase(unittest.TestCase):
    """Tests for 'network.py'."""

    def test_feed_forward(self):
        """Are values passed correctly trough the network?"""
        testinput = numpy.array([0.6, 0.8])

        testweights_layer1 = numpy.array([[0.4, 0.1],
                                          [0.5, 0.2],
                                          [0.6, 0.3]])
        testweights_layer2 = numpy.array([[0.2, 0.45, 0.3],
                                          [0.1, 0.7, 0.22]])

        testnodes_layer1 = numpy.array([0.32, 0.46, 0.6])
        testnodes_layer2 = numpy.array([0.451, 0.486])

        nn = Network(layersizes=[2, 3, 2], activation_function=LinearFunction)
        nn.layers[1].weights = testweights_layer1
        nn.layers[2].weights = testweights_layer2
        nn.query(testinput)

        numpy.testing.assert_array_almost_equal(nn.layers[1].nodes, testnodes_layer1)
        numpy.testing.assert_array_almost_equal(nn.layers[2].nodes, testnodes_layer2)

    def test_propagate_backward(self):
        """Are errors passed correctly backward trough the network?"""
        testinput = numpy.array([0.6, 0.8])
        testresults = numpy.array([0.1, 0.9])

        testweights_layer1 = numpy.array([[0.4, 0.1],
                                          [0.5, 0.2],
                                          [0.6, 0.3]])
        testweights_layer2 = numpy.array([[0.2, 0.45, 0.3],
                                          [0.1, 0.7, 0.22]])

        testerrors_layer1 = numpy.array([0.041779, 0.17541764, 0.074667])
        testerrors_layer2 = numpy.array([0.123201, 0.171396])

        nn = Network(layersizes=[2, 3, 2], activation_function=LinearFunction)
        nn.layers[1].weights = testweights_layer1
        nn.layers[2].weights = testweights_layer2
        nn.train(testinput, testresults)

        numpy.testing.assert_array_almost_equal(nn.layers[1].errors, testerrors_layer1)
        numpy.testing.assert_array_almost_equal(nn.layers[2].errors, testerrors_layer2)


if __name__ == '__main__':
    unittest.main()