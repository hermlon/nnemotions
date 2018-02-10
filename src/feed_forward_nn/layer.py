import numpy


class Layer:

    def __repr__(self):
        return 'size: ' + str(len(self.nodes)) + '\nnodes:' + str(self.nodes)


class InputLayer(Layer):

    def __init__(self, size):
        self.nodes = numpy.empty(size)
        # no implementation needed, but it is going to be called
        self.error = lambda: None

    def update(self, input):
        self.nodes = input
        self.on_update()


class HiddenLayer(Layer):

    def __init__(self, size, prev_layer, activation_function):
        self.activation_function = activation_function
        self.prev_layer = prev_layer
        self.nodes = numpy.empty(size)
        prev_layer.on_update = self.update
        self.weights = self.init_weights(size, len(prev_layer.nodes))
        # no implementation needed if not overwritten, but it is going to be called
        self.on_update = lambda: None

    # called from prev_layer when updated nodes
    def update(self):
        self.nodes = self.activation_function.normal(numpy.dot(self.weights, self.prev_layer.nodes))
        # call update of next_layer after updating own nodes
        self.on_update()

    # called by next_layer when passing errors
    def error(self, next_layer_errors):
        # self.errors = self.weights.T * next_layer_errors
        self.prev_layer.error(self.errors)

    def init_weights(self, y, x):
        # TODO: various better initialisations / negative values
        #return numpy.random.rand(y, x)
        return numpy.ones((y, x))

# TODO: Add Bias