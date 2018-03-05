class Quadratic:

    name = 'quadratic'

    @staticmethod
    def normal(a, desired):
        return (a - desired) ** 2


class Linear:

    name ='linear'

    @staticmethod
    def normal(a, desired):
        return a - desired
