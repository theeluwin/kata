# -*- coding: utf-8 -*-

import numpy as np

from word2vec import Word2Vec


if __name__ == '__main__':
    filepath = 'dl4j.txt'
    w2v = Word2Vec()
    w2v.build(filepath)
    w2v.fit(filepath)
    print(w2v.cos('home', 'house'))
