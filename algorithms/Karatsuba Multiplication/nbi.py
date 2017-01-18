# -*- coding: utf-8 -*-

import math
import functools


def NaiveBigIntFactory(base):

    if type(base) != int or base < 2:
        raise ValueError("base should be an integer, larger than 1")

    class MetaNaiveBigInt(type):

        def __str__(self):
            return 'nbi<{}>'.format(base)

        def __repr__(self):
            return "<class 'NaiveBigInt' with base {}>".format(base)

    @functools.total_ordering
    class NaiveBigInt(object, metaclass=MetaNaiveBigInt):

        def __init__(self, s=0):
            self.chain = []
            self.sign = 1
            t = type(s)
            if t == str:  # we're assuming that proper string comes in, which corresponds to the base
                if base > 16:
                    raise TypeError("initializing with string is supported to base <= 16 because my cat said so")
                lookup = '0123456789abcdef'
                if s[0] == '-':
                    self.sign = -1
                for i in range(len(s) - 1, -1, -1):
                    self.chain.append(lookup.index(s[i]))  # function `index` will raise error if there's some problem
            elif t == int:
                if s < 0:
                    self.sign = -1
                    s = -s
                while s:
                    self.chain.append(s % base)
                    s = int(s / base)
            else:
                raise TypeError("integer or string only!")
            self.compactify()

        def __int__(self):
            I = 0
            k = 1
            for value in self.chain:
                I += value * k
                k *= base
            return self.sign * I

        def __float__(self):
            return float(int(self))

        def __str__(self):
            self.compactify()
            delimiter = '' if base <= 10 else ' '
            verbose_sign = '' if self.sign == 1 else ('-' + delimiter)
            verbose_chain = delimiter.join([str(value) for value in reversed(self.chain)]) or '0'
            return verbose_sign + verbose_chain

        def __repr__(self):
            return 'nbi<{}>({})'.format(base, self.__str__())

        def __hash__(self):
            return int(self)

        def __nonzero__(self):
            self.compactify()
            return bool(self.size)

        def __eq__(self, z):
            z = NaiveBigInt.validate(z)
            if self.sign != z.sign:
                return False
            self.compactify()
            z.compactify()
            if self.size != z.size:
                return False
            for i in range(self.size - 1, -1, -1):
                if self.chain[i] != z.chain[i]:
                    return False
            return True

        def __ne__(self, z):
            z = NaiveBigInt.validate(z)
            if self.sign != z.sign:
                return True
            self.compactify()
            z.compactify()
            if self.size != z.size:
                return True
            for i in range(self.size - 1, -1, -1):
                if self.chain[i] != z.chain[i]:
                    return True
            return False

        def __lt__(self, z):
            z = NaiveBigInt.validate(z)
            if self.sign < z.sign:
                return True
            elif self.sign > z.sign:
                return False
            if self.sign == 1:
                if self.size > z.size:
                    return False
                elif self.size < z.size:
                    return True
                else:
                    for i in range(self.size - 1, -1, -1):
                        if self.chain[i] > z.chain[i]:
                            return False
                        elif self.chain[i] < z.chain[i]:
                            return True
            else:  # hard-coded and redundant, but this is faster
                if self.size > z.size:
                    return True
                elif self.size < z.size:
                    return False
                else:
                    for i in range(self.size - 1, -1, -1):
                        if self.chain[i] > z.chain[i]:
                            return True
                        elif self.chain[i] < z.chain[i]:
                            return False
            return True

        def __add__(self, z):
            z = NaiveBigInt.validate(z)
            return NaiveBigInt.add(self, z)

        def __sub__(self, z):
            z = NaiveBigInt.validate(z)
            return NaiveBigInt.sub(self, z)

        def __mul__(self, z):
            z = NaiveBigInt.validate(z)
            return NaiveBigInt.multiply(self, z)

        def __div__(self, z):
            z = NaiveBigInt.validate(z)
            # todo
            return z

        def __rmul__(self, z):
            return self * z

        def __lshift__(self, s):
            return self.shift(s)

        def __rshift__(self, s):
            return self.shift(-s)

        def __neg__(self):
            z = self.copy()
            z.sign = -1 * z.sign
            return z

        def __mod__(self, z):
            z = NaiveBigInt.validate(z)
            # todo
            return z

        def __abs__(self):
            z = self.copy()
            z.sign = 1
            return z

        def shift(self, s=0):
            if type(s) != int:
                raise TypeError("integer only1")
            z = self.copy()
            if s >= 0:
                z.chain = [0 for _ in range(s)] + z.chain
            else:
                z.chain = z.chain[-s:]
            return z

        def compactify(self):
            self.size = len(self.chain)
            cut = self.size
            for index in range(self.size - 1, -1, -1):
                if self.chain[index]:
                    break
                cut = index
            self.chain = self.chain[:cut]
            self.size = len(self.chain)
            return self

        def copy(self):
            self.compactify()
            z = NaiveBigInt()
            z.sign = self.sign
            z.chain = list(self.chain)
            z.size = self.size
            return z

        @staticmethod
        def validate(z):
            t = type(z)
            if t == int or t == str:
                z = NaiveBigInt(z)
            elif t != NaiveBigInt:
                raise TypeError("available types are: NaiveBigInt, int, str")
            return z

        @staticmethod
        def add(x, y):
            if type(x) != NaiveBigInt or type(y) != NaiveBigInt:
                raise TypeError("NaiveBigInt only!")
            if x.sign == 1 and y.sign == 1:
                return NaiveBigInt.adder(x, y)
            elif x.sign == 1 and y.sign == -1:
                y.sign = 1
                if x == y:
                    return NaiveBigInt(0)
                if x > y:
                    c = NaiveBigInt.complement(y, x.size)
                    z = x + c
                    z.chain = z.chain[:x.size]
                    z += 1
                elif x < y:
                    c = NaiveBigInt.complement(x, y.size)
                    z = y + c
                    z.chain = z.chain[:y.size]
                    z += 1
                    z.sign = -1
                y.sign = -1
                return z
            elif x.sign == -1 and y.sign == 1:
                return NaiveBigInt.add(y, x)
            else:
                z = NaiveBigInt.adder(x, y)
                z.sign = -1
                return z

        @staticmethod
        def sub(x, y):
            if type(x) != NaiveBigInt or type(y) != NaiveBigInt:
                raise TypeError("NaiveBigInt only!")
            z = y.copy()
            z.sign = -1 * z.sign
            return NaiveBigInt.add(x, z)

        @staticmethod
        def multiply(x, y):
            if type(x) != NaiveBigInt or type(y) != NaiveBigInt:
                raise TypeError("NaiveBigInt only!")
            x.compactify()
            y.compactify()
            z = NaiveBigInt()
            z.sign = x.sign * y.sign
            if not x.size or not y.size:
                return z
            if x.size == 1 and y.size == 1:
                m = x.chain[0] * y.chain[0]
                chain = [m % base]
                if m >= base:
                    chain.append(int(m / base))
                z.chain = chain
                z.compactify()
                return z
            m = int(math.ceil(max(x.size, y.size) / 2))
            c = [NaiveBigInt() for _ in range(4)]
            c[0].chain = x.chain[m:]
            c[1].chain = x.chain[:m]
            c[2].chain = y.chain[m:]
            c[3].chain = y.chain[:m]
            [d.compactify() for d in c]
            A = NaiveBigInt.multiply(c[0], c[2])
            C = NaiveBigInt.multiply(c[1], c[3])
            B = NaiveBigInt.multiply(NaiveBigInt.add(c[0], c[1]), NaiveBigInt.add(c[2], c[3]))
            B = NaiveBigInt.sub(B, A)
            B = NaiveBigInt.sub(B, C)
            z = NaiveBigInt.add(z, A.shift(2 * m))
            z = NaiveBigInt.add(z, B.shift(m))
            z = NaiveBigInt.add(z, C)
            return z.compactify()

        @staticmethod
        def adder(x, y):  # the 'adder' we know; ignores sign
            if type(x) != NaiveBigInt or type(y) != NaiveBigInt:
                raise TypeError("NaiveBigInt only!")
            x.compactify()
            y.compactify()
            if not x.size:
                return y.copy()
            if not y.size:
                return x.copy()
            z = NaiveBigInt()
            index = 0
            c = 0
            while True:
                a = x.chain[index] if index < x.size else 0
                b = y.chain[index] if index < y.size else 0
                t = a + b + c
                s = t % base
                c = int(t / base)
                z.chain.append(s)
                index += 1
                if index >= x.size and index >= y.size:
                    break
            if c:
                z.chain.append(c)
            z.compactify()
            return z

        @staticmethod
        def complement(x, size=None):
            if type(x) != NaiveBigInt:
                raise TypeError("NaiveBigInt only!")
            z = x.copy()
            if size:
                if type(size) != int:
                    raise TypeError("positive integer only!")
                if size < 1:
                    raise ValueError("positive integer only!")
                if size > z.size:
                    z.chain.extend([0 for _ in range(size - z.size)])
                else:
                    z.chain = z.chain[:size]
                z.size = size
            for index in range(z.size):
                z.chain[index] = base - z.chain[index] - 1
            return z

    return NaiveBigInt
