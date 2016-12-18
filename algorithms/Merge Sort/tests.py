# -*- coding: utf-8 -*-

import unittest
import ms


class TestMergeSort(unittest.TestCase):

    def setUp(self):
        self.a = [4, 2, 3, 1]
        self.b = [1, 4]
        self.c = [2, 3]

    def test_merge(self):
        self.assertEqual([1, 2, 3, 4], ms.merge(self.b, self.c))

    def test_sorte(self):
        self.assertEqual([1, 2, 3, 4], ms.sort(self.a))


if __name__ == '__main__':
    unittest.main()
