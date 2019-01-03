# Linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colnames = ['inputs', 'targets']
data = pd.read_csv('random_pairs.csv', names=colnames)

inputs = list(map(lambda d: d, data['inputs']))
targets = list(map(lambda d: d, data['targets']))

w1 = np.random.uniform(low=0, high=1, size=None)
b = np.random.uniform(low=0, high=1, size=None)
cost_history = []


def cost(x, y, weight, bias):
    total_error = 0.0
    total_error = (total_error + ((y - (weight*x+bias)) ** 2))/100
    return total_error


def update_weights(ins, targets, weight, bias, learning_rate):
    weight_gradient = 0
    bias_gradient = 0
    for i in range(len(ins)):
        c = cost(inputs[i], targets[i], weight, bias)
        cost_history.append(c)
        weight_gradient += -2 * ins[i]*(targets[i]-(weight * ins[i] + bias))
        bias_gradient += -2 * (targets[i]-(weight * ins[i] + bias))

    weight -= (weight_gradient/len(ins)) * learning_rate
    bias -= (bias_gradient/len(ins)) * learning_rate

    return weight, bias


def train(ins, targets, weight, bias, learning_rate, iterations):
    for i in range(iterations):
        weight, bias = update_weights(ins, targets, weight, bias, learning_rate)
    return weight, bias, iterations


weight, bias, iterations = train(inputs, targets, w1, b, 0.0001, 80000)
print(weight, bias)


plt.figure(1)
plt.axis((0, iterations, 0, 100))
plt.suptitle('Cost over time')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.plot(cost_history, 'r-')

plt.figure(2)
plt.axis((0, 110, 0, 150))
plt.figure(2)
plt.xlabel('Mock x')
plt.ylabel('Mock y')
plt.plot(inputs, targets, "b.")
x = np.linspace(0, 100, 1000)
Y = bias + (weight * x)
plt.plot(x, Y, 'g-')
plt.show()
