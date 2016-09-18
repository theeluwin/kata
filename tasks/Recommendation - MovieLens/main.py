#!/usr/bin/env python
# -*- coding: utf-8 -*-

# most of the code were borrowed from http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import sys
import csv
import codecs
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds


available_methods = ['memory_based', 'model_based']


def inting(x):
    try:
        return int(float(x))
    except:
        return None


def floating(x):
    try:
        return float(x)
    except:
        return None


def purify_row(row):
    row = row.split('\t')
    return [inting(row[0]), inting(row[1]), floating(row[2])]


def read_data(filepath):
    rows = codecs.open(filepath, 'r', encoding='utf-8').read().split('\n')
    return np.array([purify_row(row) for row in rows if row])


def data2matrix(data, users, items):
    matrix = np.zeros((users, items))
    for row in data:
        matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2]
    return matrix


def evaluate(test_matrix, pred_matrix):
    nonzeros = test_matrix.nonzero()
    test_y = test_matrix[nonzeros].flatten()
    pred_y = pred_matrix[nonzeros].flatten()
    print("rms: {}".format(np.sqrt(mean_squared_error(test_y, pred_y))))


def predict(matrix):
    similarity = pairwise_distances(matrix, metric='cosine')
    means = matrix.mean(axis=1)[:, np.newaxis]
    normalized_matrix = matrix - means
    return means + similarity.dot(normalized_matrix) / np.abs(similarity).sum(axis=1)[:, np.newaxis]


def main(method='memory_based', cheat=False):
    if method not in available_methods:
        print("available methods: {}".format(', '.join(available_methods)))
        return
    train_data = read_data('data/train.csv')
    test_data = read_data('data/test.csv')
    data = np.concatenate([train_data, test_data])
    users = len(np.unique(data[:, 0]))
    items = len(np.unique(data[:, 1]))
    if cheat:
        train_data = data
    train_matrix = data2matrix(train_data, users, items)
    test_matrix = data2matrix(test_data, users, items)
    if method == 'memory_based':
        pred_matrix = predict(train_matrix)
    elif method == 'model_based':
        u, s, vt = svds(train_matrix, k=20)
        pred_matrix = np.dot(np.dot(u, np.diag(s)), vt)
    evaluate(test_matrix, pred_matrix)


if __name__ == '__main__':
    try:
        method = sys.argv[1]
    except:
        method = 'memory_based'
    try:
        cheat = bool(sys.argv[2])
    except:
        cheat = False
    if method not in available_methods:
        print("usage: python {} [method]\navailable methods: {}".format(sys.argv[0], ', '.join(available_methods)))
    else:
        main(method, cheat)
