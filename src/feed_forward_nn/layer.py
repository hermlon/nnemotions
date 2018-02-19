import numpy


class Layer:

    def __repr__(self):
        return 'size: ' + str(len(self.nodes)) + '\nnodes:' + str(self.nodes)


class InputLayer(Layer):

    def __init__(self, size):
        self.nodes = numpy.empty(size)
        # no implementation needed
        self.error = None
        self.errors = None

    def update(self, input):
        self.nodes = input
        self.on_update()


class HiddenLayer(Layer):

    def __init__(self, size, prev_layer, activation_function, weights=None):
        self.prev_layer = prev_layer
        self.activation_function = activation_function
        self.weights = weights
        self.nodes = numpy.empty(size)
        prev_layer.on_update = self.update
        # no implementation needed if not overwritten, but it is going to be called
        self.on_update = lambda: None

        if self.weights is None:
            self.weights = self.init_weights(size, len(prev_layer.nodes))

    # called from prev_layer when updated nodes
    def update(self):
        self.nodes = self.activation_function.normal(numpy.dot(self.weights, self.prev_layer.nodes))
        # call update of next_layer after updating own nodes
        self.on_update()

    # called by next_layer when passing errors
    def error(self, layer_errors):
        self.errors = layer_errors
        # don't calculate for input layer
        if self.prev_layer.error is not None:
            prev_layer_errors = numpy.dot(self.weights.T, self.errors)
            self.prev_layer.error(prev_layer_errors)

    def init_weights(self, y, x):
        # TODO: various better initialisations / negative values
        #return numpy.random.rand(y, x)
        return numpy.ones((y, x))

# TODO: Add Bias