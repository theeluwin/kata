# -*- coding: utf-8 -*-

import time
import random

random.seed(time.time())


def partition(a, l=0, r=None):
    if r is None:
        r = len(a) - 1
    if l >= r:
        return l
    q = random.randrange(l, r + 1)
    a[l], a[q] = a[q], a[l]
    i = l + 1
    for j in range(l + 1, r + 1):
        if a[j] < a[l]:
            a[i], a[j] = a[j], a[i]
            i += 1
    q = i - 1
    if q > l:
        a[l], a[q] = a[q], a[l]
    return q


def sort(a, l=0, r=None):
    if r is None:
        r = len(a) - 1
    if l >= r:
        return
    q = partition(a, l, r)
    if q > l:
        sort(a, l, q - 1)
    if q < r:
        sort(a, q + 1, r)
