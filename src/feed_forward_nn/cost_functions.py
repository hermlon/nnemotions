class CostFunctions:

    @staticmethod
    def quadratic(a, desired):
        return (a - desired) ** 2

    @staticmethod
    def linear(a, desired):
        return a - desired
