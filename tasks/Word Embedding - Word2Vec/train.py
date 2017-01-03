# -*- coding: utf-8 -*-

import numpy as np

from word2vec import Word2Vec


def cos(v1, v2):
    norm = np.linalg.norm
    return v1.dot(v2) / norm(v1) / norm(v2)


if __name__ == '__main__':
    filepath = 'dl4j.txt'
    w2v = Word2Vec()
    w2v.build(filepath)
    w2v.fit(filepath)
    v1 = w2v.ivectors['home']
    v2 = w2v.ivectors['house']
    print(cos(v1, v2))
