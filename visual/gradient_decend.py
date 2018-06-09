from nnemotions.network.feed_forward_nn import FeedForwardNN
from nnemotions.network.nn_functions import SigmoidActivationFunction, QuadraticCostFunction, ReLuActivationFunction, LinearActivationFunction, LinearCostFunction
import random
import numpy as np
import matplotlib.pyplot as plt

nn = FeedForwardNN(layersizes=[1, 1], learningrate=0.01, activation_function=LinearActivationFunction, cost_function=QuadraticCostFunction)

costs = []
for epoch in range(10000):
    input = np.array(random.randint(0, 100), ndmin=2) / 100
    des_output = np.array(input * 2, ndmin=2) / 100
    nn.train(input, des_output)
    costs.append(nn.cost)

plt.plot(costs)
plt.show()