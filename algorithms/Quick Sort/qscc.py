# -*- coding: utf-8 -*-


class QSCC(object):

    methods = ['first', 'final', 'median']

    def __init__(self, a, method='first'):
        if method not in self.methods:
            raise NotImplementedError("available methods are {}".format(', '.join(self.methods)))
        self.a = a
        self.m = 0
        self.method = method

    def partition(self, l=0, r=None):
        if r is None:
            r = len(self.a) - 1
        if l >= r:
            return l
        self.m += r - l
        if self.method == 'first':
            pass
        elif self.method == 'final':
            self.a[l], self.a[r] = self.a[r], self.a[l]
        elif self.method == 'median':
            q = l + int((r - l) / 2)
            if self.a[l] > self.a[q]:
                if self.a[q] > self.a[r]:  # l > q > r
                    self.a[l], self.a[q] = self.a[q], self.a[l]
                else:
                    if self.a[l] > self.a[r]:  # l > r > q
                        self.a[l], self.a[r] = self.a[r], self.a[l]
                    else:  # r > l > q
                        pass
            else:
                if self.a[q] > self.a[r]:
                    if self.a[l] > self.a[r]:  # q > l > r
                        pass
                    else:  # q > r > l
                        self.a[l], self.a[r] = self.a[r], self.a[l]
                else:  # r > q > l
                    self.a[l], self.a[q] = self.a[q], self.a[l]
        p = self.a[l]
        i = l + 1
        for j in range(l + 1, r + 1):
            if self.a[j] < p:
                self.a[i], self.a[j] = self.a[j], self.a[i]
                i += 1
        q = i - 1
        if q > l:
            self.a[l], self.a[q] = self.a[q], self.a[l]
        return q

    def sort(self, l=0, r=None):
        if r is None:
            r = len(self.a) - 1
        if l >= r:
            return
        q = self.partition(l, r)
        if q > l:
            self.sort(l, q - 1)
        if q < r:
            self.sort(q + 1, r)
