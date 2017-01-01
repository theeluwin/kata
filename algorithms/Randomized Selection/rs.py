# -*- coding: utf-8 -*-

import time
import random

random.seed(time.time())


def partition(a):
    q = random.randrange(0, len(a))
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


def select(a, i):  # note that this selects i'th largest element, not the index!
    if len(a) == 1:
        return a[0]
    a = list(a)
    q = partition(a)
    j = i - 1
    if q == j:
        return a[q]
    elif q < j:
        return select(a[q + 1:], j - q)
    else:
        return select(a[:q], j + 1)
