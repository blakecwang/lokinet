#!/usr/bin/env python3

# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import math

class LokiNet:
    def __init__(self, x, y):
        self.input    = x
        self.weights1 = np.random.rand(self.input.shape[1], 4) 
        self.weights2 = np.random.rand(4, 1)                 
        self.y        = y
        self.output   = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def sigmoid(self, x):
      return 1 / (1 + math.exp(-x))

x = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
y = np.array(
    [10,
     11,
     12]
)
loki = LokiNet(x, y)
loki.feedforward()
