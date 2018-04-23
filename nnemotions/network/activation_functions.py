import numpy


class SigmoidFunction:

    name = 'sigmoid'

    @staticmethod
    def normal(x):
        #return x
        return 1.0 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(sigmoidx):
        return sigmoidx * (1 - sigmoidx)


class LinearFunction:

    name = 'linear'

    @staticmethod
    def normal(x):
        return x

    @staticmethod
    def derivative(x):
        return 1


class ReLuFunction:

    name = 'relu'

    @staticmethod
    def normal(x):
        x[x<0] = 0
        return x

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


activation_functions = {
    SigmoidFunction.name: SigmoidFunction,
    LinearFunction.name: LinearFunction,
    ReLuFunction.name: ReLuFunction
}