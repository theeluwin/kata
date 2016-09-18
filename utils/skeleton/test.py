#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import unittest

from scenario import Scenario


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario(cheat=False, verbose=False)

    def test_baseline(self):
        self.assertEqual(1, self.scenario.baseline())


if __name__ == '__main__':
    unittest.main()
