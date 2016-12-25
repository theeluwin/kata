# -*- coding: utf-8 -*-

import unittest

from nbi import NaiveBigIntFactory


class TestNaiveBigInt(unittest.TestCase):

    def setUp(self):
        self.nbi = NaiveBigIntFactory(10)
        self.x = self.nbi(12345)
        self.y = self.nbi(6789)

    def test_todo(self):  # todo
        pass

    def test_coursera(self):
        nbi = NaiveBigIntFactory(10)
        a = nbi('3141592653589793238462643383279502884197169399375105820974944592')
        b = nbi('2718281828459045235360287471352662497757247093699959574966967627')
        c = nbi.multiply(a, b)
        self.assertEqual('8539734222673567065463550869546574495034888535765114961879601127067743044893204848617875072216249073013374895871952806582723184', str(c))


if __name__ == '__main__':
    unittest.main()
