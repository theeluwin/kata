# -*- coding: utf-8 -*-

import qs
import qsc
import codecs
import unittest

from qscc import QSCC


class TestQuickSort(unittest.TestCase):

    def setUp(self):
        self.a = [3, 8, 2, 5, 1, 4, 7, 6]
        self.b = [3, 8, 2, 5, 1, 4, 7, 6]

    def reader(self):
        file = codecs.open('data.txt', 'r', encoding='utf-8')
        raw = file.read().strip().replace('\r', '')
        data = [int(value) for value in raw.split('\n')]
        file.close()
        return data

    def test_partition(self):
        q = qs.partition(self.a)
        for i in range(0, q):
            self.assertLess(self.a[i], self.a[q])
        for i in range(q + 1, len(self.a)):
            self.assertGreater(self.a[i], self.a[q])

    def test_sort(self):
        qs.sort(self.b)
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], self.b)

    def test_coursera(self):
        self.assertEqual(162085, qsc.sort(self.reader(), method='first'))
        self.assertEqual(164123, qsc.sort(self.reader(), method='final'))
        self.assertEqual(138382, qsc.sort(self.reader(), method='median'))


if __name__ == '__main__':
    unittest.main()
