# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division

import unittest
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(f):
    return f * (1 - f)


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.epsilon = 1e-6
        self.x = np.array([[1, 2], [-1, -2]])
        self.sx = np.array([[0.73105858, 0.88079708], [0.26894142, 0.11920292]])
        self.dsx = np.array([[0.19661193, 0.10499359], [0.19661193, 0.10499359]])

    def inormdiff(self, v1, v2):
        return np.amax(np.fabs(v1 - v2))

    def test_sigmoid(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(sigmoid(self.x), self.sx))

    def test_sigmoid_grad(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(sigmoid_grad(sigmoid(self.x)), self.dsx))


if __name__ == '__main__':
    unittest.main()
