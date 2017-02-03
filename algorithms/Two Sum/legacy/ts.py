# -*- coding: utf-8 -*-


def twosum(a, h, t):
    for i in a:
        if t - i in h:
            return True
    return False


def hasher(a):
    h = {}
    for i in a:
        h[i] = True
    return h
