#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import unittest

from scenario import Scenario

#------------------------------------------------------------------------------#

import cf
import numpy as np


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario(cheat=False, verbose=False)
        self.scenario.n_user = 2
        self.scenario.n_item = 2

    def test_read_data(self):
        train_data = self.scenario.read_data('data/train.csv')
        test_data = self.scenario.read_data('data/test.csv')
        self.assertEqual((75000, 3), train_data.shape)
        self.assertEqual((25000, 3), test_data.shape)

    def test_data2matrix(self):
        data = [(1, 1, 1), (1, 2, 1), (2, 1, 1)]
        matrix = self.scenario.data2matrix(data)
        self.assertEqual(0, (matrix - np.array([[1, 1], [1, 0]])).sum())

    def test_evaluate(self):
        test_matrix = np.array([[1, 1], [1, 0]])
        pred_matrix = np.array([[0, 0], [0, 1]])
        self.assertEqual(1, self.scenario.evaluate(test_matrix, pred_matrix))


class CFTest(unittest.TestCase):

    def setUp(self):
        self.matrix = np.array([[1, 0], [0, 1], [0, 0]])
        self.n_user = 3
        self.n_item = 2

    def test_model_based(self):
        pred_matrix = cf.model_based(self.matrix, k=1)
        self.assertEqual(0, pred_matrix[self.n_user - 1, :].sum())

    def test_memory_based(self):
        pred_matrix = cf.memory_based(self.matrix)
        self.assertEqual(0, pred_matrix[self.n_user - 1, :].sum())

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    if len(sys.argv) > 1:
        Scenario.benchmark(Scenario)
    else:
        unittest.main()
