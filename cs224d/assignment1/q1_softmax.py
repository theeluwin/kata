# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import unittest
import numpy as np


def softmax(x):
    try:
        z = np.exp(x - x.max(axis=1)[:, np.newaxis])
        return z / z.sum(axis=1)[:, np.newaxis]
    except:
        z = np.exp(x - x.max())
        return z / z.sum()


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.epsilon = 1e-6
        self.x1 = np.array([1, 2])
        self.y1 = np.array([0.26894142, 0.73105858])
        self.x2 = np.array([[1001, 1002], [3, 5]])
        self.y2 = np.array([[0.26894142, 0.73105858], [0.11920292, 0.88079708]])
        self.x3 = np.array([[-1001, -1002]])
        self.y3 = np.array([0.73105858, 0.26894142])

    def inormdiff(self, v1, v2):
        return np.amax(np.fabs(v1 - v2))

    def test_softmax_1d(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(softmax(self.x1), self.y1))

    def test_softmax_nd(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(softmax(self.x2), self.y2))

    def test_softmax_nd_minus(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(softmax(self.x3), self.y3))


if __name__ == '__main__':
    unittest.main()
