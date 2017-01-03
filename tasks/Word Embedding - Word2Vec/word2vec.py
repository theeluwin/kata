# -*- coding: utf-8 -*-

import codecs
import pickle
import numpy as np


class Word2Vec(object):

    def __init__(self, dimension=100, alpha=1e-5, window=5, verbose=True):
        self.dimension = dimension
        self.alpha = alpha
        self.window = window
        self.verbose = verbose
        self.ivectors = {}
        self.ovectors = {}
        self.vocabulary = set()

    def log(self, message):
        if self.verbose:
            print(message)

    def build(self, filepath):
        self.log("building vocabulary...")
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = [token for token in line.split() if token]
                for word in sentence:
                    self.vocabulary.add(word)
        for word in self.vocabulary:
            self.ivectors[word] = np.random.normal(size=self.dimension)
            self.ovectors[word] = np.random.normal(size=self.dimension)

    def fit(self, filepath, epoch=3):
        self.log("learning...")
        for era in range(1, epoch + 1):
            step = 0
            with codecs.open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    sentence = [token for token in line.split() if token and token in self.vocabulary]
                    if len(sentence) < 2:
                        continue
                    losses = []
                    for i in range(len(sentence)):
                        iword, contexts = self.skipgram(sentence, i)
                        loss = self.learn(iword, contexts)
                        if loss and not np.isnan(loss):
                            losses.append(loss)
                    step += 1
                    if len(losses):
                        loss = sum(losses) / len(losses)
                        self.log("epoch {}, step {}, loss: {}".format(era, step, loss))
            self.checkpoint(era)

    def checkpoint(self, era):
        with open('models/epoch-{}'.format(era), 'wb') as file:
            pickle.dump(self.ivectors, file)

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[i - self.window: i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, left + right

    def learn(self, iword, contexts):
        ivector = self.ivectors[iword]
        cerrors = {}
        Z = 0
        for word in self.vocabulary:
            Z += np.exp(ivector.dot(self.ovectors[word]))
        for word in self.vocabulary:
            cerrors[word] = []
            for context in contexts:
                t = 1 if context == word else 0
                cerrors[word].append(np.exp(ivector.dot(self.ovectors[word])) / Z - t)
        for word in self.vocabulary:
            error = sum(cerrors[word])
            self.ivectors[iword] -= self.alpha * error * self.ovectors[word]
            self.ovectors[word] -= self.alpha * error * ivector
        ivector = self.ivectors[iword]
        Z = 0
        for word in self.vocabulary:
            Z += np.exp(ivector.dot(self.ovectors[word]))
        p = []
        for context in contexts:
            p.append(np.exp(ivector.dot(self.ovectors[context])) / Z)
        p = np.array(p)
        if len(p):
            return -(np.log(p)).sum() / len(p)
        else:
            return np.nan
