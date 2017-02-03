# -*- coding: utf-8 -*-

import codecs
import unittest

from ts import twosum, hasher


class TestTwoSum(unittest.TestCase):

    def setUp(self):
        self.a = [-6, -4, -2, 0, 2, 4, 6]

    def reader(self, filename):
        file = codecs.open(filename, 'r', encoding='utf-8')
        data = [int(line) for line in file]
        file.close()
        return data

    def test_twosum(self):
        h = hasher(self.a)
        self.assertEqual(True, twosum(self.a, h, 0))
        self.assertEqual(False, twosum(self.a, h, 1))

    def test_coursera(self):  # this is not actually a test...
        data = self.reader('data.txt')
        h = hasher(data)
        a = h.keys()
        T = [t for t in range(-10000, 10001)]
        count = {t: 0 for t in T}
        yet = {t: True for t in T}
        for i in a:
            left = list(yet.keys())
            for t in left:
                if t - i in h:
                    count[t] = 1
                    del yet[t]
        print(sum(count.values()))  # ???


if __name__ == '__main__':
    unittest.main()
