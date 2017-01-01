# -*- coding: utf-8 -*-

import sys
import math

sys.path.append('../Merge Sort')

import ms  # tested to be valid


def partition(a, q):
    a[0], a[q] = a[q], a[0]
    i = 1
    for j in range(1, len(a)):
        if a[j] < a[0]:
            a[i], a[j] = a[j], a[i]
            i += 1
    q = i - 1
    if q > 0:
        a[0], a[q] = a[q], a[0]
    return q


def medianth(n):
    if n % 2:
        return int(n / 2)
    else:
        return int((n - 1) / 2)


def select(a, i):  # note that this selects i'th largest element, not the index!
    if len(a) == 1:
        return a[0]
    a = list(a)
    reverse = {}  # this is actually a naive trick, but still works in O(n)!
    for j in range(len(a)):
        reverse[a[j]] = j
    m = []
    for j in range(int(math.ceil(len(a) / 5))):
        c = ms.sort(a[5 * j: 5 * (j + 1)])
        m.append(c[medianth(len(c))])
    p = select(m, medianth(len(m)) + 1)
    q = partition(a, reverse[p])
    j = i - 1
    if q == j:
        return a[q]
    elif q < j:
        return select(a[q + 1:], j - q)
    else:
        return select(a[:q], j + 1)
