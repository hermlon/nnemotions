import numpy
from feed_forward_nn.layer import InputLayer, HiddenLayer
from feed_forward_nn.activation_functions import SigmoidFunction


class Network:

    def __init__(self, layersizes=[5, 3, 2, 2]):
        self.layers = []
        self.layers.append(InputLayer(layersizes[0]))
        # Every layersize except the first one as it already is the input layer
        for layersize in layersizes[1:]:
            # set prev_layer to the last element in self.layers
            self.layers.append(HiddenLayer(layersize, self.layers[-1], SigmoidFunction))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]


        # Testing

        self.input_layer.update(numpy.array([0.7, 0.7, 0.7, 0.7, 0.7]))

        for layer in self.layers:
            print(repr(layer))

        #self.output_layer.error(None)

Network()