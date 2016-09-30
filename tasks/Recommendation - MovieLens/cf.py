#!/usr/bin/env python
# -*- coding: utf-8 -*-

# most of the code were borrowed from http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2

from __future__ import division, print_function, unicode_literals

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds


def model_based(matrix, k=20):
    matrix = matrix.astype('d')
    u, s, vt = svds(matrix, k)
    return np.dot(np.dot(u, np.diag(s)), vt)


def memory_based(matrix):
    matrix = matrix.astype('d')
    similarity = pairwise_distances(matrix, metric='cosine')
    means = matrix.mean(axis=1)[:, np.newaxis]
    normalized_matrix = matrix - means
    return means + similarity.dot(normalized_matrix) / np.abs(similarity).sum(axis=1)[:, np.newaxis]
