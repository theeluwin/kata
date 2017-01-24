# -*- coding: utf-8 -*-

from __future__ import division

import random
import unittest
import numpy as np

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params
from cs224d.data_utils import *


def getSentenceFeature(tokens, wordVectors, sentence):
    return np.mean(wordVectors[[tokens[word] for word in sentence], :], axis=0)


def softmaxRegression(features, labels, weights, regularization=0.0, nopredictions=False):
    m = features.shape[0] if len(features.shape) > 1 else 1
    prob = softmax(features.dot(weights))
    cost = np.mean(-np.log(prob[np.arange(m), labels]), axis=0) + regularization * np.sum(np.power(weights, 2)) / 2
    pred = np.argmax(prob, axis=1)
    prob[np.arange(m), labels] -= 1
    grad = regularization * weights + features.T.dot(prob) / m
    return (cost, grad) if nopredictions else (cost, grad, pred)


def accuracy(y, yhat):
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def softmax_wrapper(features, labels, weights, regularization=0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights, regularization)
    return cost, grad


class TestSoftmaxRegression(unittest.TestCase):

    def setUp(self):
        num_class = 5
        batch_size = 10
        dataset = StanfordSentiment()
        tokens = dataset.tokens()
        num_vocab = len(tokens)
        _, params, _ = load_saved_params()
        vectors = params[:num_vocab, :] + params[num_vocab:, :]
        dim = vectors.shape[-1]
        self.weights = 0.1 * np.random.randn(dim, num_class)
        self.features = np.zeros((batch_size, dim))
        self.labels = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            words, self.labels[i] = dataset.getRandomTrainSentence()
            self.features[i, :] = getSentenceFeature(tokens, vectors, words)

    def grad_and_cost(self):
        return lambda weights: softmaxRegression(self.features, self.labels, weights, 1.0, nopredictions=True)

    def test_softmax_regression(self):
        self.assertEqual(True, gradcheck_naive(self.grad_and_cost(), self.weights))


if __name__ == '__main__':
    unittest.main()
