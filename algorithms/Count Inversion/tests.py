# -*- coding: utf-8 -*-

import ci
import codecs
import unittest


class TestCountInversion(unittest.TestCase):

    def setUp(self):
        self.a = [1, 3, 5]
        self.b = [2, 4, 6]
        self.c = [1, 3, 5, 2, 4, 6]

    def reader(self):
        file = codecs.open('data.txt', 'r', encoding='utf-8')
        raw = file.read().strip().replace('\r', '')
        data = [int(value) for value in raw.split('\n')]
        file.close()
        return data

    def test_merge_and_count(self):
        c, z = ci.merge_and_count(self.a, self.b)
        self.assertEqual([1, 2, 3, 4, 5, 6], c)
        self.assertEqual(z, 3)

    def test_count_inversion(self):
        c, z = ci.count_inversion(self.c)
        self.assertEqual([1, 2, 3, 4, 5, 6], c)
        self.assertEqual(3, z)

    def test_coursera(self):
        c, z = ci.count_inversion(self.reader())
        self.assertEqual(2407905288, z)


if __name__ == '__main__':
    unittest.main()
