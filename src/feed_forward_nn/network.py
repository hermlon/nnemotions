import numpy
from feed_forward_nn.layer import InputLayer, HiddenLayer
from feed_forward_nn.activation_functions import SigmoidFunction
from feed_forward_nn.cost_functions import CostFunctions


class Network:

    def __init__(self, layersizes=[2, 2, 2], activation_function=SigmoidFunction):
        self.layers = []
        self.layers.append(InputLayer(layersizes[0]))
        # Every layersize except the first one as it already is the input layer
        for layersize in layersizes[1:]:
            # set prev_layer to the last element in self.layers
            self.layers.append(HiddenLayer(layersize, self.layers[-1], activation_function))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def query(self, input):
        self.input_layer.update(input)

    def train(self, input, desired_output, learninrate=0.5):
        self.input_layer.update(input)
        error = CostFunctions.quadratic(self.output_layer.nodes, desired_output)
        self.output_layer.error(error)


"""
nn = Network()
nn.train(numpy.array([0.7, 0.7]), numpy.array([0.2, 0.2]))
for layer in nn.layers:
    print(layer.errors)"""

# warum ist der Fehler in schicht 2 grosser als in schicht 3?

# TODO: Write tests for everything