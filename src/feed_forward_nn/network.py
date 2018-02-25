import numpy
from feed_forward_nn.layer import InputLayer, HiddenLayer, OutputLayer
from feed_forward_nn.activation_functions import SigmoidFunction
from feed_forward_nn.cost_functions import CostFunctions


class Network:

    def __init__(self, layersizes=[2, 2, 2], activation_function=SigmoidFunction):
        self.layers = []
        self.cost = None

        self.layers.append(InputLayer(layersizes[0]))
        # Every layersize except the first and the last one
        for layersize in layersizes[1:-1]:
            # set prev_layer to the last element in self.layers
            layer = HiddenLayer(layersize, self.layers[-1], activation_function)
            self.layers.append(layer)
        self.layers.append(OutputLayer(layersizes[-1], self.layers[-1], activation_function))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def query(self, input):
        self.input_layer.start_forward_pass(input)

        return self.output_layer.nodes

    def train(self, input, desired_output, learninrate=1):
        self.query(input)
        error = CostFunctions.quadratic(self.output_layer.nodes, desired_output)
        self.output_layer.start_backward_pass(error, learninrate)

        self.cost = CostFunctions.quadratic(self.output_layer.nodes, desired_output).sum(axis=0)
        return self.output_layer.nodes
