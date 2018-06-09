from nnemotions.network.layer import InputLayer, HiddenLayer, OutputLayer
from nnemotions.network.nn_functions import SigmoidActivationFunction, QuadraticCostFunction


class FeedForwardNN:

    def __init__(self, layersizes=[2, 2, 2], activation_function=SigmoidActivationFunction, cost_function=QuadraticCostFunction, learningrate=1, bias=False):
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

    def train(self, input, desired_output, learninrate=None, cost_function=None, update_weights=True):
        if learninrate is None:
            learninrate = self.learningrate
        if cost_function is None:
            cost_function = self.cost_function

        self.query(input)
        error = desired_output - self.output_layer.nodes
        if update_weights:
            self.output_layer.start_backward_pass(error, learninrate)

        #print('.........')
        #print(self.output_layer.nodes)
        # cost as float instead of one element numpy array
        self.cost = float(cost_function.normal(self.output_layer.nodes, desired_output).sum(axis=0))
        return self.output_layer.nodes
