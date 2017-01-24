# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import random
import unittest
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def ffnn(data, labels, params, dimensions):
    m = data.shape[0]
    offset = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[offset: offset + Dx * H], (Dx, H))
    offset += Dx * H
    b1 = np.reshape(params[offset: offset + H], (1, H))
    offset += H
    W2 = np.reshape(params[offset: offset + H * Dy], (H, Dy))
    offset += H * Dy
    b2 = np.reshape(params[offset: offset + Dy], (1, Dy))
    a1 = data
    z2 = a1.dot(W1) + b1
    a2 = sigmoid(z2)
    z3 = a2.dot(W2) + b2
    y = softmax(z3)
    cost = -np.log(y[labels.astype(np.bool)]).sum() / m
    d3 = y - labels
    gradW2 = a2.T.dot(d3) / m
    gradb2 = d3.sum(axis=0) / m
    d2 = d3.dot(W2.T) * sigmoid_grad(a2)
    gradW1 = a1.T.dot(d2) / m
    gradb1 = d2.sum(axis=0) / m
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    return cost, grad, y


def forward_backward_prop(data, labels, params, dimensions):
    cost, grad, _ = ffnn(data, labels, params, dimensions)
    return cost, grad


def iris_test():
    iris = datasets.load_iris()
    Dx = iris.data.shape[-1]
    H = 2
    Dy = iris.target_names.shape[-1]
    dimensions = [Dx, H, Dy]
    m = len(iris.data)
    labels = np.zeros((m, Dy))
    labels[np.arange(m), iris.target] = 1
    params = np.random.randn((Dx + 1) * H + (H + 1) * Dy)
    assert gradcheck_naive(lambda params: forward_backward_prop(iris.data, labels, params, [Dx, H, Dy]), params)
    threshold = 1e-6
    alpha = 1e-1
    costs = []
    while (len(costs) < 2000) or (costs[-2] - costs[-1] > threshold):
        cost, grad = forward_backward_prop(iris.data, labels, params, dimensions)
        costs.append(cost)
        if not len(costs) % 1000:
            print("epoch {}, cost: {}".format(len(costs), costs[-1]))
        params = params - alpha * grad
    print("done")
    _, _, y = ffnn(iris.data, labels, params, dimensions)
    pred = np.argmax(y, axis=1)
    accuracy = np.count_nonzero(iris.target == pred) / np.size(pred)
    print("accuracy: %.2f" % accuracy)
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.plot(np.arange(len(costs)), costs)
    plt.show()


class TestNeural(unittest.TestCase):

    def setUp(self):
        N = 20
        dimensions = [10, 5, 10]
        data = np.random.randn(N, dimensions[0])
        labels = np.zeros((N, dimensions[-1]))
        for i in range(N):
            labels[i, random.randint(0, dimensions[-1] - 1)] = 1
        self.params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )
        self.cost_and_grad = lambda params: forward_backward_prop(data, labels, params, dimensions)

    def test_forward_backward_prop(self):
        self.assertEqual(True, gradcheck_naive(self.cost_and_grad, self.params))


if __name__ == '__main__':
    unittest.main()
