from feed_forward_nn.layer import InputLayer, HiddenLayer, OutputLayer
from feed_forward_nn.activation_functions import SigmoidFunction
from feed_forward_nn.cost_functions import Quadratic


class Network:

    def __init__(self, layersizes=[2, 2, 2], activation_function=SigmoidFunction, cost_function=Quadratic, learningrate=1, bias=False):
        self.layersizes = layersizes
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.learningrate = learningrate
        self.bias = bias

        self.layers = []
        self.cost = None

        self.layers.append(InputLayer(layersizes[0], bias=bias))
        # Every layersize except the first and the last one
        for layersize in layersizes[1:-1]:
            # set prev_layer to the last element in self.layers
            layer = HiddenLayer(layersize, self.layers[-1], activation_function, bias=bias)
            self.layers.append(layer)
        self.layers.append(OutputLayer(layersizes[-1], self.layers[-1], activation_function))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def query(self, input):
        self.input_layer.start_forward_pass(input)

        return self.output_layer.nodes

    def train(self, input, desired_output, learninrate=None, cost_function=None):
        if learninrate is None:
            learninrate = self.learningrate
        if cost_function is None:
            cost_function = self.cost_function

        self.query(input)
        error = cost_function.normal(self.output_layer.nodes, desired_output)
        self.output_layer.start_backward_pass(error, learninrate)

        self.cost = cost_function.normal(self.output_layer.nodes, desired_output).sum(axis=0)
        return self.output_layer.nodes
