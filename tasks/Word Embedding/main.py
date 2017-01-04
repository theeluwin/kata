# -*- coding: utf-8 -*-

import numpy as np

from word2vec import Word2Vec


def train_word2vec():
    trainpath = 'dl4j.txt'
    w2v = Word2Vec()
    w2v.build(trainpath)
    w2v.fit(trainpath)
    w2v.evaluate()


if __name__ == '__main__':
    train_word2vec()
