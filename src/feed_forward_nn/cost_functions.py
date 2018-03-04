class Quadratic:

    @staticmethod
    def normal(a, desired):
        return (a - desired) ** 2

    def __repr__(self):
        return 'quadratic'


class Linear:

    @staticmethod
    def normal(a, desired):
        return a - desired

    def __repr__(self):
        return 'linear'
