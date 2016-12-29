# -*- coding: utf-8 -*-

import rs
import unittest


class TestRandomizedSelection(unittest.TestCase):

    def setUp(self):
        self.a = [3, 8, 2, 5, 1, 4, 7, 6]

    def test_select(self):
        self.assertEqual(3, rs.select(self.a, 3))
        self.assertEqual(7, rs.select(self.a, 7))


if __name__ == '__main__':
    unittest.main()
