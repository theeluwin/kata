# -*- coding: utf-8 -*-

import sys
import codecs
import unittest

sys.path.append('../Karatsuba Multiplication')

from heap import Heap
from nbi import NaiveBigIntFactory


class TestHeap(unittest.TestCase):

    def setUp(self):
        self.nbi = NaiveBigIntFactory(2)
        self.ints = [4, 3, 2, 1]
        self.nbis = [self.nbi(4), self.nbi(3), self.nbi(2), self.nbi(1)]

    def test_push_int(self):
        heap = Heap(self.ints)
        self.assertEqual([1, 2, 3, 4], heap.tree)

    def test_push_nbi(self):
        heap = Heap(self.nbis)
        self.assertEqual([self.nbi(1), self.nbi(2), self.nbi(3), self.nbi(4)], heap.tree)

    def test_pop_int(self):
        heap = Heap(self.ints)
        self.assertEqual(1, heap.pop())
        self.assertEqual(2, heap.pop())
        self.assertEqual(3, heap.pop())
        self.assertEqual(4, heap.pop())

    def test_pop_nbi(self):
        heap = Heap(self.nbis)
        self.assertEqual(self.nbi(1), heap.pop())
        self.assertEqual(self.nbi(2), heap.pop())
        self.assertEqual(self.nbi(3), heap.pop())
        self.assertEqual(self.nbi(4), heap.pop())

    def test_coursera(self):
        n = 10000
        m = 0
        low = Heap(compare=lambda x, y: x > y)
        high = Heap(compare=lambda x, y: x < y)
        with codecs.open('data.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if not line:
                    continue
                x = int(line)
                if not len(low.tree):
                    low.push(x)
                else:
                    if x < low.tree[0]:
                        low.push(x)
                    else:
                        high.push(x)
                while len(high.tree) > len(low.tree):
                    low.push(high.pop())
                while len(low.tree) > len(high.tree) + 1:
                    high.push(low.pop())
                if len(low.tree) < len(high.tree):
                    y = high.tree[0]
                else:
                    y = low.tree[0]
                m = (m + y) % n
        self.assertEqual(1213, m)


if __name__ == '__main__':
    unittest.main()
