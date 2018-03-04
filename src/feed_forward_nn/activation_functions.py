import numpy


class SigmoidFunction:

    @staticmethod
    def normal(x):
        #return x
        return 1.0 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(sigmoidx):
        return sigmoidx * (1 - sigmoidx)

    def __repr__(self):
        return 'sigmoid'


class LinearFunction:

    @staticmethod
    def normal(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

    def __repr__(self):
        return 'linear'


class ReLuFunction:

    @staticmethod
    def normal(x):
        x[x<0] = 0
        return x

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    def __repr__(self):
        return 'relu'