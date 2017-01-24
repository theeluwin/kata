# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import random
import unittest
import numpy as np


def gradcheck_naive(cost_and_grad, x, h=1e-4, epsilon=1e-5, verbose=False):
    state = random.getstate()  # reset random state to fixed one! omg! this WAS important!
    random.setstate(state)
    cost, grad = cost_and_grad(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] += h
        random.setstate(state)
        cost_right, _ = cost_and_grad(x)
        x[ix] -= 2 * h
        random.setstate(state)
        cost_left, _ = cost_and_grad(x)
        x[ix] += h
        numgrad = (cost_right - cost_left) / h / 2
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > epsilon:
            if verbose:
                print("Gradient check failed.")
                print("First gradient error found at index %s" % str(ix))
                print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return False
        it.iternext()
    if verbose:
        print("Gradient check succeeded.")
    return True


class TestGradcheckNaive(unittest.TestCase):

    def setUp(self):
        self.cost_and_grad = lambda x: (np.sum(np.power(x, 2)), x * 2)
        self.x_scalar = np.array(123.456)
        self.x_1d = np.random.randn(3,)
        self.x_2d = np.random.randn(4, 5)

    def test_gradcheck_naive_scalar(self):
        self.assertEqual(True, gradcheck_naive(self.cost_and_grad, self.x_scalar))

    def test_gradcheck_naive_1d(self):
        self.assertEqual(True, gradcheck_naive(self.cost_and_grad, self.x_1d))

    def test_gradcheck_naive_2d(self):
        self.assertEqual(True, gradcheck_naive(self.cost_and_grad, self.x_2d))


if __name__ == '__main__':
    unittest.main()
