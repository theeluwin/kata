#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys

from os.path import dirname, abspath

sys.path.append((lambda f, x: f(f(f(x))))(dirname, abspath(__file__)))

from utils.draft import Draft

#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd


class Scenario(Draft):

    available_methods = ['baseline']

    def __init__(self, **kwargs):
        self.header = ['value', 'class']
        super(Scenario, self).__init__(**kwargs)

    def read_data(self, filename):
        df = pd.read_csv(filename, sep=',', names=self.header)
        return df.values

    def evaluate(self, test_y, pred_y):
        diff = test_y - pred_y
        accuracy = len(diff[diff == 0]) / len(diff)
        if self.verbose:
            print("accuracy: {}".format(Draft.verbose_percent(accuracy)))
        return accuracy

    def baseline(self, train_data):
        ub = train_data[train_data[:, -1] == 0][:, 0].max()
        lb = train_data[train_data[:, -1] == 1][:, 0].min()
        threshold = (ub + lb) / 2
        return np.vectorize(lambda x: 1 if x > threshold else 0)

    @Draft.print_elapsed
    def method_baseline(self):
        return self.baseline(self.train_data)

    @Draft.print_elapsed
    def load_data(self):
        self.train_data = self.read_data('data/train.csv')
        self.test_data = self.read_data('data/test.csv')
        if self.cheat:
            self.train_data = np.concatenate([self.train_data, self.test_data])
        self.train_x, self.train_y = np.split(self.train_data, [-1], axis=1)
        self.test_x, self.test_y = np.split(self.test_data, [-1], axis=1)

    @Draft.print_elapsed
    def play(self):
        self.load_data()
        predict = getattr(self, 'method_{}'.format(self.method))()
        pred_y = predict(self.test_x)
        return self.evaluate(self.test_y, pred_y)

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    scenario = Scenario(sys_argv=sys.argv)
    scenario.play()
