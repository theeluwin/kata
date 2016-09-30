#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import numpy as np

from collections import defaultdict


class NaiveBayesClassifier:

    def __init__(self, dimension):
        self.dimension = dimension
        self.classes_y = np.array([])
        self.classes_x = [np.array([]) for k in range(self.dimension)]
        self.c_prevalance = defaultdict(lambda: 0)
        self.c_bayes = defaultdict(lambda: [defaultdict(lambda: 0) for k in range(self.dimension)])
        self.l_prevalance = defaultdict(lambda: -np.inf)
        self.l_bayes = defaultdict(lambda: [defaultdict(lambda: -np.inf) for k in range(self.dimension)])

    def fit(self, train_x, train_y):
        self.classes_y = np.unique(np.concatenate([self.classes_y, train_y]))
        self.classes_x = [np.unique(np.concatenate([self.classes_x[k], train_x[:, k]])) for k in range(self.dimension)]
        for class_y in self.classes_y:
            self.c_prevalance[class_y] += len(train_y[train_y == class_y])
        total_y = sum(self.c_prevalance.values())
        for class_y in self.classes_y:
            self.l_prevalance[class_y] = np.log(self.c_prevalance[class_y] / total_y)
        for class_y in self.classes_y:
            conditional = np.array([train_x[i] for i in range(len(train_x)) if train_y[i] == class_y])
            for k in range(self.dimension):
                for class_x in self.classes_x[k]:
                    self.c_bayes[class_y][k][class_x] += len(conditional[conditional[:, k] == class_x])
                total_x = sum(self.c_bayes[class_y][k].values())
                for class_x in self.classes_x[k]:
                    self.l_bayes[class_y][k][class_x] = np.log(self.c_bayes[class_y][k][class_x] / total_x)

    def predict(self, test_x):
        pred_y = []
        for x in test_x:
            likelihood = lambda class_y: self.l_prevalance[class_y] + sum([self.l_bayes[class_y][k][x[k]] for k in range(self.dimension)])
            pred_y.append(max(self.classes_y, key=likelihood))
        return np.array(pred_y)
