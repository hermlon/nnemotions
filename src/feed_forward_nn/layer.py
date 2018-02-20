import numpy


class Layer:

    def __init__(self, size):
        self.prev_layer = None
        self.next_layer = None
        self.errors = None
        self.nodes = numpy.empty(size)

    def pass_forward(self):
        if self.next_layer is not None:
            self.next_layer.pass_forward()

    def pass_backward(self):
        if self.prev_layer is not None:
            self.prev_layer.pass_backward()

    def __repr__(self):
        return 'size: ' + str(len(self.nodes)) + '\nnodes:' + str(self.nodes)


class InputLayer(Layer):

    def start_forward_pass(self, inputnodes):
        self.nodes = inputnodes
        self.pass_forward()


class HiddenLayer(Layer):

    def __init__(self, size, prev_layer, activation_function, weights=None):
        super().__init__(size)

        self.prev_layer = prev_layer
        self.prev_layer.next_layer = self

        self.activation_function = activation_function
        self.weights = weights

        if self.weights is None:
            self.init_weights(size, len(self.prev_layer.nodes))

    def pass_forward(self):
        self.nodes = self.activation_function.normal(numpy.dot(self.weights, self.prev_layer.nodes))
        super().pass_forward()

    def pass_backward(self):
        self.errors = numpy.dot(self.next_layer.weights.T, self.next_layer.errors)
        super().pass_backward()

    def init_weights(self, y, x):
        # TODO: various better initialisations / negative values
        #return numpy.random.rand(y, x)
        self.weights = numpy.ones((y, x))


class OutputLayer(HiddenLayer):

    def start_backward_pass(self, errors):
        self.errors = errors
        # these are the errors, don't calculate anything, just call prev_layer
        # seems to be the best way, instead of reimplementing the pass_backward class of layer
        # because the one in HiddenLayer is overridden
        Layer.pass_backward(self)

# TODO: Add Bias