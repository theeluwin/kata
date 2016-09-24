#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys

from os.path import dirname, abspath

sys.path.append((lambda f, x: f(f(f(x))))(dirname, abspath(__file__)))

from utils.draft import Draft

#------------------------------------------------------------------------------#

import csv
import codecs
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout


class Scenario(Draft):

    def __init__(self, **kwargs):
        self.class2index = {
            'setosa': 0,
            'versicolor': 1,
            'virginica': 2,
        }
        self.index2class = ['setosa', 'versicolor', 'virginica']
        self.ohe = OneHotEncoder()
        self.available_methods = ['keras']
        super(Scenario, self).__init__(**kwargs)

    def read_data(self, filepath):
        lines = codecs.open(filepath, 'r', encoding='utf-8').read().strip().split('\n')
        data = np.array([row for row in csv.reader(lines)])
        x = data[:, :-1].astype(np.float)
        y = np.array([self.class2index[datum] for datum in data[:, -1]])[:, np.newaxis]
        return x, y

    def flush(self, filepath, pred_y):
        result = codecs.open(filepath, 'w', encoding='utf-8')
        result.write('\n'.join([self.index2class[int(datum)] for datum in pred_y.flatten()]) + '\n')
        result.close()

    def evaluate(self, test_y, pred_y):
        diff = test_y.flatten() - pred_y.flatten()
        accuracy = len(diff[diff == 0]) / len(diff)
        if self.verbose:
            print("accuracy: %5.02f%%" % (accuracy * 100))
        return accuracy

    @Draft.print_elapsed
    def keras(self, train_x, ohe_y):
        samples, order_x = train_x.shape
        samples, order_y = ohe_y.shape
        model = Sequential()
        model.add(Dense(8, activation='relu', init='normal', input_dim=order_x))
        model.add(Dense(8, activation='relu', init='normal'))
        model.add(Dense(order_y, activation='softmax', init='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.fit(train_x, ohe_y, nb_epoch=200, batch_size=5, verbose=2)
        return model

    @Draft.print_elapsed
    def play(self):
        train_x, train_y = self.read_data('data/train.csv')
        test_x, test_y = self.read_data('data/test.csv')
        ohe_y = self.ohe.fit_transform(train_y).toarray()
        if self.method == 'keras':
            model = self.keras(train_x, ohe_y)
            pred_y = model.predict_classes(test_x, verbose=0)
        self.flush('output_predict_keras.txt', pred_y)
        self.evaluate(test_y, pred_y)

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    scenario = Scenario(sys_argv=sys.argv)
    scenario.play()
