#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import unittest

from scenario import Scenario

#------------------------------------------------------------------------------#

import numpy as np


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario(cheat=False, verbose=False)

    def test_read_data(self):
        train_x, train_y = self.scenario.read_data('data/train.csv')
        test_x, test_y = self.scenario.read_data('data/test.csv')
        self.assertEqual(train_x.shape, (120, 4))
        self.assertEqual(train_y.shape, (120, 1))
        self.assertEqual(test_x.shape, (30, 4))
        self.assertEqual(test_y.shape, (30, 1))

    def test_evaluate(self):
        test_y = np.array([0, 1, 1, 2])
        pred_y = np.array([0, 0, 1, 2])
        self.assertEqual(0.75, self.scenario.evaluate(test_y, pred_y))

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    unittest.main()
