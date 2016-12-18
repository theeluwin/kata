# -*- coding: utf-8 -*-


def merge_and_count(a, b):
    z = 0
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
            z += len_a - i
    if i == len_a:
        while j < len_b:
            c.append(b[j])
            j += 1
    if j == len_b:
        while i < len_a:
            c.append(a[i])
            i += 1
    return c, z


def count_inversion(c):
    len_c = len(c)
    if len_c == 1:
        return c, 0
    k = int(len_c / 2)
    a = c[:k]
    b = c[k:]
    a, x = count_inversion(a)
    b, y = count_inversion(b)
    c, z = merge_and_count(a, b)
    return c, x + y + z
