import numpy


class SigmoidFunction:

    @staticmethod
    def normal(x):
        #return x
        return 1.0 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(x):
        return SigmoidFunction.normal(x) * (1 - SigmoidFunction.normal(x))
