# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import glob
import random
import unittest
import numpy as np
import os.path as op
import cPickle as pickle

SAVE_PARAMS_EVERY = 1000


def load_saved_params():
    genesis = 0
    for file in glob.glob('saved_params_*.npy'):
        epoch = int(op.splitext(op.basename(file))[0].split('_')[2])
        if (epoch > genesis):
            genesis = epoch
    if genesis > 0:
        with open('saved_params_%d.npy' % genesis, 'r') as file:
            params = pickle.load(file)
            state = pickle.load(file)
        return genesis, params, state
    else:
        return genesis, None, None


def save_params(epoch, params):
    with open('saved_params_%d.npy' % epoch, 'w') as file:
        pickle.dump(params, file)
        pickle.dump(random.getstate(), file)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10, ANNEAL_EVERY=20000, verbose=True):
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    x = x0
    if not postprocessing:
        postprocessing = lambda x: x
    expcost = None
    for iter in xrange(start_iter + 1, iterations + 1):
        cost, grad = f(x)
        x = postprocessing(x - step * grad)
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            if verbose:
                print("iter %d: %f" % (iter, expcost))
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    return x


class TestSGD(unittest.TestCase):

    def setUp(self):
        self.quad = lambda x: (np.sum(np.power(x, 2)), x * 2)
        self.epsilon = 1e-6
        self.alpha = 1e-2
        self.epoch = 1000
        self.every = 100
        self.initial_right = 0.5
        self.initial_center = 0.0
        self.initial_left = -1.5

    def inormdiff(self, v1, v2):
        return np.amax(np.fabs(v1 - v2))

    def test_sgd_initial_right(self):
        theta = sgd(self.quad, self.initial_right, self.alpha, self.epoch, PRINT_EVERY=self.every, verbose=False)
        self.assertGreaterEqual(self.epsilon, self.inormdiff(theta, 0))

    def test_sgd_initial_center(self):
        theta = sgd(self.quad, self.initial_center, self.alpha, self.epoch, PRINT_EVERY=self.every, verbose=False)
        self.assertGreaterEqual(self.epsilon, self.inormdiff(theta, 0))

    def test_sgd_initial_left(self):
        theta = sgd(self.quad, self.initial_left, self.alpha, self.epoch, PRINT_EVERY=self.every, verbose=False)
        self.assertGreaterEqual(self.epsilon, self.inormdiff(theta, 0))


if __name__ == '__main__':
    unittest.main()
