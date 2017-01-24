# -*- coding: utf-8 -*-

import random
import unittest
import numpy as np

from q1_softmax import softmax
from q2_sigmoid import sigmoid
from q2_gradcheck import gradcheck_naive


def normalizeRows(x):
    return x / np.linalg.norm(x, axis=1)[:, np.newaxis]


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    y = softmax(outputVectors.dot(predicted))
    cost = -np.log(y[target])
    y[target] -= 1
    gradPred = outputVectors.T.dot(y)
    grad = predicted * y[:, np.newaxis]
    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    indicies = [target]
    for _ in range(K):
        index = dataset.sampleTokenIdx()
        while index == target:
            index = dataset.sampleTokenIdx()
        indicies.append(index)
    labels = np.array([1] + [-1 for _ in range(K)])
    vectors = outputVectors[indicies, :]
    likelies = sigmoid(labels * vectors.dot(predicted))
    cost = -np.sum(np.log(likelies))
    delta = labels * (likelies - 1)
    gradPred = delta.dot(vectors)
    grad = np.zeros(outputVectors.shape)
    temps = predicted * delta[:, np.newaxis]
    for k in range(K + 1):
        grad[indicies[k], :] += temps[k]
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    center_index = tokens[currentWord]
    center_vector = inputVectors[center_index, :]
    for context in contextWords:
        cost_context, grad_center, gradOut_context = word2vecCostAndGradient(center_vector, tokens[context], outputVectors, dataset)
        cost += cost_context
        gradIn[center_index, :] += grad_center
        gradOut += gradOut_context
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    indices = [tokens[context] for context in contextWords]
    chunk_vector = np.sum([inputVectors[index, :] for index in indices], axis=0)
    cost, grad_chunk, gradOut = word2vecCostAndGradient(chunk_vector, tokens[currentWord], outputVectors, dataset)
    for index in indices:
        gradIn[index, :] += grad_chunk
    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        denom = 1
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom
    return cost, grad


class TestWord2Vec(unittest.TestCase):

    def setUp(self):
        self.epsilon = 1e-6
        self.x = np.array([[3.0, 4.0], [1, 2]])
        self.nx = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
        self.window_size = 5
        self.vocabulary_size = 5
        self.tokens = dict([(chr(ord('a') + i), i) for i in range(self.vocabulary_size)])
        self.vocabulary = self.tokens.keys()
        self.dataset = type('dummy', (), {})()
        self.dataset.sampleTokenIdx = self.dummySampleTokenIdx
        self.dataset.getRandomContext = self.getRandomContext
        self.vectors = normalizeRows(np.random.randn(2 * self.vocabulary_size, 3))

    def inormdiff(self, v1, v2):
        return np.amax(np.fabs(v1 - v2))

    def dummySampleTokenIdx(self):
        return random.randint(0, self.vocabulary_size - 1)

    def getRandomContext(self, window):
        center = self.vocabulary[self.dummySampleTokenIdx()]
        context = [self.vocabulary[self.dummySampleTokenIdx()] for _ in range(2 * window)]
        return center, context

    def word2vec(self, method, cng):
        return lambda vectors: word2vec_sgd_wrapper(method, self.tokens, vectors, self.dataset, self.window_size, cng)

    def test_normalize_rows(self):
        self.assertGreaterEqual(self.epsilon, self.inormdiff(normalizeRows(self.x), self.nx))

    def test_skipgram_softmax(self):
        self.assertEqual(True, gradcheck_naive(self.word2vec(skipgram, softmaxCostAndGradient), self.vectors))

    def test_skipgram_negative_sampling(self):
        self.assertEqual(True, gradcheck_naive(self.word2vec(skipgram, negSamplingCostAndGradient), self.vectors))

    def test_cbow_softmax(self):
        self.assertEqual(True, gradcheck_naive(self.word2vec(cbow, softmaxCostAndGradient), self.vectors))

    def test_cbow_negative_sampling(self):
        self.assertEqual(True, gradcheck_naive(self.word2vec(cbow, negSamplingCostAndGradient), self.vectors))


if __name__ == '__main__':
    unittest.main()
