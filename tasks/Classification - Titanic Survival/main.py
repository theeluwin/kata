#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import sys
import csv
import codecs
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from nbc import NaiveBayesClassifier


available_methods = ['naive_bayes', 'random_forest']


def read_data(filepath):
    raws = codecs.open(filepath, 'r', encoding='utf-8').read().strip().split('\n')
    rows = csv.reader(raws)
    x, y = [], []
    for row in rows:
        try:
            pclass = int(row[0])
            suvived = int(row[1])
            sex = 0 if row[3] == 'male' else 1
            age = int(min(int(float(row[4])), 40) / 10) if row[4] else -1
            cabin = int(row[9][-1]) % 2 if row[9] and row[9][-1].isdigit() else 3
        except:
            continue
        x.append([pclass, sex, age, cabin])
        y.append(suvived)
    return np.array(x), np.array(y)


def flush(filepath, pred_y):
    result = codecs.open(filepath, 'w', encoding='utf-8')
    result.write('\n'.join(np.char.mod('%d', pred_y)) + '\n')
    result.close()


def evaluate(test_y, pred_y):
    xor = np.logical_xor(test_y, pred_y)
    accuracy = len(xor[xor == False]) / len(xor)
    print("accuracy: {}".format(int(accuracy * 10000) / 100.0))


def main(method='naive_bayes', cheat=False):
    if method not in available_methods:
        print("available methods: {}".format(', '.join(available_methods)))
        return
    train_x, train_y = read_data('data/train.csv')
    test_x, test_y = read_data('data/test.csv')
    if method == 'naive_bayes':
        nbc = NaiveBayesClassifier(train_x.shape[1])
        nbc.fit(train_x, train_y)
        if cheat:
            nbc.fit(test_x, test_y)
        pred_y = nbc.predict(test_x)
        flush('output_naive_bayes.txt', pred_y)
    elif method == 'random_forest':
        rfc = RandomForestClassifier()
        rfc.fit(train_x, train_y)
        if cheat:
            rfc.fit(test_x, test_y)
        pred_y = rfc.predict(test_x)
        flush('output_random_forest.txt', pred_y)
    evaluate(test_y, pred_y)


if __name__ == '__main__':
    try:
        method = sys.argv[1]
    except:
        method = 'naive_bayes'
    try:
        cheat = bool(sys.argv[2])
    except:
        cheat = False
    if method not in available_methods:
        print("usage: python {} [method]\navailable methods: {}".format(sys.argv[0], ', '.join(available_methods)))
    else:
        main(method, cheat)
