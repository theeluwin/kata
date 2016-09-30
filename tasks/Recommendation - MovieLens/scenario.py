#!/usr/bin/env python
# -*- coding: utf-8 -*-

# most of the code were borrowed from http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2

from __future__ import division, print_function, unicode_literals

import sys

from os.path import dirname, abspath

sys.path.append((lambda f, x: f(f(f(x))))(dirname, abspath(__file__)))

from utils.draft import Draft

#------------------------------------------------------------------------------#

import cf
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


class Scenario(Draft):

    available_methods = ['model_based', 'memory_based']

    def __init__(self, **kwargs):
        self.header = ['user_id', 'item_id', 'rating', 'timestamp']
        super(Scenario, self).__init__(**kwargs)

    def read_data(self, filename):
        df = pd.read_csv(filename, sep='\t', names=self.header)
        data = df.values[:, :-1]
        return data

    def data2matrix(self, data):
        matrix = np.zeros((self.n_user, self.n_item))
        for user_id, item_id, rating in data:
            matrix[user_id - 1, item_id - 1] = rating
        return matrix

    def evaluate(self, test_matrix, pred_matrix):
        nonzeros = test_matrix.nonzero()
        test_data = test_matrix[nonzeros].flatten()
        pred_data = pred_matrix[nonzeros].flatten()
        rms = np.sqrt(mean_squared_error(test_data, pred_data))
        if self.verbose:
            print("rms: {}".format(rms))
        return rms

    @Draft.print_elapsed
    def method_model_based(self):
        return cf.model_based(self.train_matrix, k=20)

    @Draft.print_elapsed
    def method_memory_based(self):
        return cf.memory_based(self.train_matrix)

    @Draft.print_elapsed
    def load_data(self):
        train_data = self.read_data('data/train.csv')
        test_data = self.read_data('data/test.csv')
        data = np.concatenate([train_data, test_data])
        self.n_user = len(np.unique(data[:, 0]))
        self.n_item = len(np.unique(data[:, 1]))
        if self.cheat:
            train_data = data
        self.train_matrix = self.data2matrix(train_data)
        self.test_matrix = self.data2matrix(test_data)

    @Draft.print_elapsed
    def play(self):
        self.load_data()
        pred_matrix = getattr(self, 'method_{}'.format(self.method))()
        return self.evaluate(self.test_matrix, pred_matrix)

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    scenario = Scenario(sys_argv=sys.argv)
    scenario.play()
