# -*- coding: utf-8 -*-


def merge(a, b):
    c = []
    i, j = 0, 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    if i == len_a:
        while j < len_b:
            c.append(b[j])
            j += 1
    if j == len_b:
        while i < len_a:
            c.append(a[i])
            i += 1
    return c


def sort(c):
    len_c = len(c)
    if len_c == 1:
        return c
    k = int(len_c / 2)
    a = sort(c[:k])
    b = sort(c[k:])
    return merge(a, b)
