# -*- coding: utf-8 -*-


def partition(a, l=0, r=None):
    if r is None:
        r = len(a) - 1
    if l == r:
        return a
    p = a[l]
    i = l + 1
    for j in range(l + 1, r + 1):
        if a[j] < p:
            a[i], a[j] = a[j], a[i]
            i += 1
    q = i - 1
    a[l], a[q] = a[q], a[l]
    return q


def sort(a, l=0, r=None):
    if r is None:
        r = len(a) - 1
    if l == r:
        return a
    if r - l == 1:
        if a[r] < a[l]:
            a[r], a[l] = a[l], a[r]
        return a
    q = partition(a, l, r)
    if q > l:
        sort(a, l, q - 1)
    if q < r:
        sort(a, q + 1, r)
    return a
