#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import unittest

from scenario import Scenario

#------------------------------------------------------------------------------#

import numpy as np


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario(cheat=False, verbose=False)

    def test_read_data(self):
        train_data = self.scenario.read_data('data/train.csv')
        test_data = self.scenario.read_data('data/test.csv')
        self.assertEqual((10, 2), train_data.shape)
        self.assertEqual((5, 2), test_data.shape)

    def test_evaluate(self):
        test_y = np.array([1, 0, 0, 0])
        pred_y = np.array([0, 1, 0, 0])
        self.assertEqual(0.5, self.scenario.evaluate(test_y, pred_y))

    def test_baseline(self):
        train_data = np.array([[0, 0], [10, 1]])
        predict = self.scenario.baseline(train_data)
        self.assertEqual(0, predict(3))
        self.assertEqual(1, predict(7))

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    if len(sys.argv) > 1:
        Scenario.benchmark(Scenario)
    else:
        unittest.main()
