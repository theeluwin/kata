# -*- coding: utf-8 -*-

import ds
import unittest


class TestDeterministicSelection(unittest.TestCase):

    def setUp(self):
        self.a = [3, 8, 2, 5, 1, 4, 7, 6, 7]  # the test cast should include duplicated values, since we're using the reverse trick

    def test_medianth(self):
        self.assertEqual(0, ds.medianth(2))
        self.assertEqual(1, ds.medianth(3))

    def test_partition(self):
        self.assertEqual(2, ds.partition(list(self.a), 0))

    def test_select(self):
        self.assertEqual(3, ds.select(self.a, 3))
        self.assertEqual(7, ds.select(self.a, 7))


if __name__ == '__main__':
    unittest.main()
