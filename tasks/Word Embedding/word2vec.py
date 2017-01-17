# -*- coding: utf-8 -*-

import csv
import codecs
import pickle
import numpy as np


class Word2Vec(object):

    def __init__(self, dimension=50, alpha=1e-1, window=5, every=1000, outpath='output', goldpath='gold.txt', verbose=True):
        self.dimension = dimension
        self.alpha = alpha
        self.window = window
        self.every = every
        self.verbose = verbose
        self.ivectors = {}
        self.ovectors = {}
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
        v1 = self.ivectors[w1]
        v2 = self.ivectors[w2]
        return v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

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

    def fit(self, filepath, epoch=1):
        self.log("learning...")
        for era in range(1, epoch + 1):
            step = 0
            with codecs.open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    sentence = [token for token in line.split() if token and token in self.vocabulary]
                    if len(sentence) < 2:
                        continue
                    befores = []
                    afters = []
                    for i in range(len(sentence)):
                        iword, contexts = self.skipgram(sentence, i)
                        before, after = self.learn(iword, contexts)
                        if before and not np.isnan(before) and after and not np.isnan(after):
                            befores.append(before)
                            afters.append(after)
                    step += 1
                    if not step % self.every:
                        self.checkpoint(era, step)
                    if len(befores) and len(afters):
                        before = sum(befores) / len(befores)
                        after = sum(afters) / len(afters)
                        if not step % int(self.every / 10):
                            self.log("epoch %d, step %d, before loss: %7.4f, after loss: %7.4f" % (era, step, before, after))
        self.checkpoint(era, step)
        self.log("done.")

    def checkpoint(self, era, step):
        self.evaluate()
        with open('{}/word2vec-epoch-{}-step-{}'.format(self.outpath, era, step), 'wb') as file:
            pickle.dump(self, file)

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[i - self.window: i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, left + right

    def cost(self, iword, contexts):
        ivector = self.ivectors[iword]
        Z = sum([np.exp(ivector.dot(self.ovectors[word])) for word in self.vocabulary])
        p = np.array([np.exp(ivector.dot(self.ovectors[context])) / Z for context in contexts])
        if len(p):
            return -(np.log(p)).sum() / len(p)
        else:
            return np.nan

    def learn(self, iword, contexts):
        before = self.cost(iword, contexts)
        ivector = np.array(self.ivectors[iword])
        cerrors = {}
        Z = sum([np.exp(ivector.dot(self.ovectors[word])) for word in self.vocabulary])
        for word in self.vocabulary:
            cerrors[word] = []
            for context in contexts:
                t = 1 if context == word else 0
                cerrors[word].append(np.exp(ivector.dot(self.ovectors[word])) / Z - t)
        for word in self.vocabulary:
            error = sum(cerrors[word])
            self.ivectors[iword] -= self.alpha * error * self.ovectors[word]
            self.ovectors[word] -= self.alpha * error * ivector
        after = self.cost(iword, contexts)
        return before, after
