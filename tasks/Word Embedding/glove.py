# -*- coding: utf-8 -*-

import os
import csv
import codecs
import pickle
import numpy as np


class Glove(object):

    def __init__(self, dimension=50, smoothing=0.75, alpha=1e-2, window=5, batch=100, outpath='output', goldpath='gold.txt', verbose=True):
        self.dimension = dimension
        self.smoothing = smoothing
        self.alpha = alpha
        self.window = window
        self.batch = batch
        self.verbose = verbose
        self.vector_f = {}
        self.vector_r = {}
        self.bias_f = {}
        self.bias_r = {}
        self.X = {}
        self.logX = {}
        self.fX = {}
        self.contexts_of = {}
        self.having_context = {}
        self.vectors = {}
        self.vocabulary = set()
        self.outpath = outpath
        self.goldpath = goldpath

    def log(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    def evaluate(self):
        golds = []
        with codecs.open(self.goldpath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                golds.append({
                    'v1': row[0],
                    'v2': row[1],
                    'sim': float(row[2]),
                })
        x = np.array([gold['sim'] for gold in golds])
        y = []
        for gold in golds:
            y.append(self.cos(gold['v1'], gold['v2']))
        y = np.array(y)
        p = x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)
        self.log("score: %4.02f" % p)
        return p

    def cos(self, w1, w2):
        v1 = self.vectors[w1] if w1 in self.vectors else (self.vector_f[w1] + self.vector_r[w1]) / 2
        v2 = self.vectors[w2] if w2 in self.vectors else (self.vector_f[w2] + self.vector_r[w2]) / 2
        return v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    def build(self, filepath):
        self.log("building vocabulary...")
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = [token for token in line.split() if token]
                for word in sentence:
                    self.vocabulary.add(word)
        for word in self.vocabulary:
            self.vector_f[word] = np.random.normal(size=self.dimension)
            self.vector_r[word] = np.random.normal(size=self.dimension)
            self.bias_f[word] = np.random.normal()
            self.bias_r[word] = np.random.normal()
            self.contexts_of[word] = set()
            self.having_context[word] = set()
        self.log("done")

    def fit(self, filepath, epoch=100):
        self.log("building co-occurence matrix...")
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = [token for token in line.split() if token and token in self.vocabulary]
                len_sentence = len(sentence)
                for i in range(len_sentence):
                    iword = sentence[i]
                    for j in range(max(0, i - self.window), min(len_sentence, i + self.window + 1)):
                        if i == j:
                            continue
                        jword = sentence[j]
                        score = 1 / abs(i - j)
                        pair = (iword, jword)
                        if pair in self.X:
                            self.X[pair] += score
                        else:
                            self.X[pair] = score
                        self.contexts_of[iword].add(jword)
                        self.having_context[jword].add(iword)
        self.log("done")
        self.log("analyzing co-occurence matrix...")
        cat = 0
        for pair in self.X:
            cat = max(cat, self.X[pair])
            self.logX[pair] = np.log(self.X[pair])
        for pair in self.X:
            dog = self.X[pair]
            self.fX[pair] = 1 if dog >= cat else np.power(dog / cat, self.smoothing)
        self.log("done")
        self.log("learning...")
        vocabulary = list(self.vocabulary)
        len_vocabulary = len(vocabulary)
        for era in range(1, epoch + 1):
            steps = np.arange(0, len_vocabulary, self.batch)
            for step in steps:
                words = vocabulary[step: step + self.batch]
                round_bias_f = {}
                round_bias_r = {}
                round_vector_f = {}
                round_vector_r = {}
                for word in words:
                    bias_f = 0
                    vector_f = np.zeros(self.dimension)
                    for context in self.contexts_of[word]:
                        bias_f += self.fX[(word, context)] * self.calculate_error(word, context)
                        vector_f += self.fX[(word, context)] * self.calculate_error(word, context) * self.vector_r[context]
                    round_bias_f[word] = bias_f
                    round_vector_f[word] = vector_f
                    bias_r = 0
                    vector_r = np.zeros(self.dimension)
                    for acontext in self.having_context[word]:
                        bias_r += self.fX[(acontext, word)] * self.calculate_error(acontext, word)
                        vector_r += self.fX[(acontext, word)] * self.calculate_error(acontext, word) * self.vector_f[acontext]
                    round_bias_r[word] = bias_r
                    round_vector_r[word] = vector_r
                for word in words:
                    self.bias_f[word] -= self.alpha * round_bias_f[word]
                    self.bias_r[word] -= self.alpha * round_bias_r[word]
                    self.vector_f[word] -= self.alpha * round_vector_f[word]
                    self.vector_r[word] -= self.alpha * round_vector_r[word]
                self.log("epoch %d, step %d, current loss: %7.4f" % (era, step + len(words), self.calculate_loss()))
            if not era % 10:
                self.checkpoint(era)
        self.log("done")
        self.log("adjusting word vectors...")
        for word in self.vocabulary:
            self.vectors[word] = (self.vector_f[word] + self.vector_r[word]) / 2
        self.log("done")

    def checkpoint(self, era):
        self.evaluate()
        with open('{}/glove-epoch'.format(self.outpath, era), 'wb') as file:
            pickle.dump(self, file)

    def calculate_error(self, iword, jword):
        return self.vector_f[iword].dot(self.vector_r[jword]) + self.bias_f[iword] + self.bias_r[jword] - self.logX[(iword, jword)]

    def calculate_loss(self):
        J = 0
        for iword, jword in self.X:
            error = self.calculate_error(iword, jword)
            J += self.fX[(iword, jword)] * error * error
        return J / 2
