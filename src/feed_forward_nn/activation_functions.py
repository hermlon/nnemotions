import numpy


class SigmoidFunction:

    @staticmethod
    def normal(x):
        #return x
        return 1.0 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(sigmoidx):
        return sigmoidx * (1 - sigmoidx)


class LinearFunction:

    @staticmethod
    def normal(x):
        return x

    @staticmethod
    def derivative(x):
        return 1
