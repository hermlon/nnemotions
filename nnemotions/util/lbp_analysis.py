from nnemotions.network.feed_forward_nn import FeedForwardNN
from nnemotions.detection.emotion.local_binary_pattern import BinaryPatternAnalysis
import numpy


class LBPAnalysis:

    def __init__(self, nn_config):
        self.nn = FeedForwardNN(layersizes=nn_config.layersizes, activation_function=nn_config.activation_function,
                                cost_function=nn_config.cost_function, learningrate=nn_config.learningrate, bias=nn_config.bias)
        self.blocksize = (nn_config.blocksize, nn_config.blocksize)

    def query(self, input, output=None):
        bpa = BinaryPatternAnalysis(input, self.blocksize)
        # convert list to numpy array, maybe outsource to get_histogram()
        lbp_histogram = numpy.array(bpa.get_histogram(), ndmin=2).T
        # normalize to numbers from 0 to 1
        lbp_histogram = lbp_histogram / lbp_histogram.max()

        if output is not None:
            result = self.nn.train(lbp_histogram, output)
        else:
            result = self.nn.query(lbp_histogram)
        return result
