# Linear regression

import numpy as np
import matplotlib.pyplot as plt

# Sample data in x,y pairs
data = [[2, 1],
 [3, 1],
 [2, 0.5],
 [1, 1],
 [3, 1.5],
 [3.5, 0.5],
 [4, 1.5],
 [5.5, 1],
 [4.5, 2],
 [5, 0.5]
 ]


# Initialize random weights
rand_slope = np.random.uniform(low=0, high=1, size=None)
rand_bias = np.random.uniform(low=0, high=1, size=None)
cost_history = []

def cost(input_data, slope, bias):
    data_size = len(input_data)
    total_cost = 0
    for i in range(data_size):
        total_cost += (input_data[i][1] - (slope*input_data[i][0] + bias))**2
    return total_cost/data_size

def update_weights(in_slope, in_bias, dataset, learning_rate):
    d_cost_d_slope = 0
    d_cost_d_bias = 0
    data_size = len(dataset)

    # Calculate partials
    # -2x(y - (mx + b))
    for i in range(data_size):
        cost = (dataset[i][1] - (in_slope * dataset[i][0] + in_bias)) ** 2
        cost_history.append(cost)
        # Calculate partial derivatives
        # -2x_i(y_i - (mx_i + b))
        d_cost_d_slope += -2 * dataset[i][0] * (dataset[i][1] - (in_slope * dataset[i][0] + in_bias))

        # -2(y_i - (mx_i + b))
        d_cost_d_bias += -2 * (dataset[i][1] - (in_slope * dataset[i][0] + in_bias))
    in_slope -= (d_cost_d_slope / data_size) * learning_rate
    in_bias -= (d_cost_d_bias / data_size) * learning_rate

    return in_slope, in_bias


def train(slope, bias, dataset, learning_rate, iterations):
    for i in range(iterations):
       slope, bias = update_weights(slope, bias, dataset, learning_rate)
    return slope, bias, iterations


slope, bias, iterations = train(rand_slope, rand_bias, data, 0.001, 5000)

# Regression
plt.figure(1)
plt.plot(list(map(lambda d: d[0], data)), list(map(lambda d: d[1], data)), 'bo')
plt.axis([0,5,0,6])
plt.ylabel("y")
plt.xlabel("x")
axes = plt.gca()
T = np.linspace(0, 500, 1100)
Y = bias + (slope * T)
plt.plot(T, Y, 'g-')

# Cost
plt.figure(2)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0, iterations, 0, 3])
plt.plot(cost_history, 'r-')
plt.show()
