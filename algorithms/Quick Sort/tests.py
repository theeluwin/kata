# -*- coding: utf-8 -*-

import qs
import unittest


class TestQuickSort(unittest.TestCase):

    def setUp(self):
        self.a = [3, 8, 2, 5, 1, 4, 7, 6]
        self.b = [3, 8, 2, 5, 1, 4, 7, 6]

    def test_partition(self):
        qs.partition(self.a)
        self.assertEqual([1, 2, 3, 5, 8, 4, 7, 6], self.a)

    def test_sort(self):
        qs.sort(self.b)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], self.b)


if __name__ == '__main__':
    unittest.main()
