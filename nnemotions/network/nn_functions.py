import numpy


class SigmoidActivationFunction:

    name = 'sigmoid'

    @staticmethod
    def normal(x):
        return 1.0 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(sigmoidx):
        return sigmoidx * (1 - sigmoidx)


class LinearActivationFunction:

    name = 'linear'

    @staticmethod
    def normal(x):
        return x

    @staticmethod
    def derivative(x):
        return 1


class ReLuActivationFunction:

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

class QuadraticCostFunction:

    name = 'quadratic'

    @staticmethod
    def normal(a, desired):
        return (a - desired) ** 2


class LinearCostFunction:

    name ='linear'

    @staticmethod
    def normal(a, desired):
        return a-desired#abs(a - desired)#


nn_functions = {
    SigmoidActivationFunction.name: SigmoidActivationFunction,
    LinearActivationFunction.name: LinearActivationFunction,
    ReLuActivationFunction.name: ReLuActivationFunction,
    QuadraticCostFunction.name: QuadraticCostFunction,
    LinearCostFunction.name: LinearCostFunction
}