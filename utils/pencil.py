# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import numpy as np


def levenshtein(s, t, trace=False, verbose=False):
    arrows = ['↑', '←', '↖']
    n, m = len(s), len(t)
    if not n:
        return m
    if not m:
        return n
    d = np.zeros((n + 1, m + 1))
    p = np.zeros((n, m))
    d[:, 0] = np.linspace(0, n, n + 1)
    d[0, :] = np.linspace(0, m, m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.array([d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1 - int(s[i - 1] == t[j - 1])])
            k = c.argmin()
            d[i, j] = c[k]
            p[i - 1, j - 1] = k
    distance = int(d[n, m])
    if verbose:
        print('\n ' + t)
        for i in range(n):
            print(s[i] + ''.join([arrows[int(k)] for k in p[i]]))
    if trace:
        return distance, p
    return distance
