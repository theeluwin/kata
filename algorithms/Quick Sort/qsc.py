# -*- coding: utf-8 -*-


def partition(a, l=0, r=None, method='first'):
    if r is None:
        r = len(a) - 1
    if l >= r:
        return l
    if method == 'first':
        pass
    elif method == 'final':
        a[l], a[r] = a[r], a[l]
    elif method == 'median':
        q = l + int((r - l) / 2)
        if a[l] > a[q]:
            if a[q] > a[r]:  # l > q > r
                a[l], a[q] = a[q], a[l]
            else:
                if a[l] > a[r]:  # l > r > q
                    a[l], a[r] = a[r], a[l]
                else:  # r > l > q
                    pass
        else:
            if a[q] > a[r]:
                if a[l] > a[r]:  # q > l > r
                    pass
                else:  # q > r > l
                    a[l], a[r] = a[r], a[l]
            else:  # r > q > l
                a[l], a[q] = a[q], a[l]
    p = a[l]
    i = l + 1
    for j in range(l + 1, r + 1):
        if a[j] < p:
            a[i], a[j] = a[j], a[i]
            i += 1
    q = i - 1
    if q > l:
        a[l], a[q] = a[q], a[l]
    return q


def sort(a, l=0, r=None, method='first'):
    if r is None:
        r = len(a) - 1
    if l >= r:
        return 0
    q = partition(a, l, r, method=method)
    m = r - l
    if q > l:
        m += sort(a, l, q - 1, method=method)
    if q < r:
        m += sort(a, q + 1, r, method=method)
    return m
