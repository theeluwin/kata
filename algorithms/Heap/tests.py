# -*- coding: utf-8 -*-

import sys
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


if __name__ == '__main__':
    unittest.main()
