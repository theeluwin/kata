#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import unittest

from draft import Draft
from pencil import levenshtein


class TooShortScenario(Draft):

    def __init__(self, **kwargs):
        self.available_methods = []
        super(TooShortScenario, self).__init__(**kwargs)


class WrongTypeScenario(Draft):

    def __init__(self, **kwargs):
        self.available_methods = None
        super(WrongTypeScenario, self).__init__(**kwargs)


class NotPlayingScenario(Draft):

    def __init__(self, **kwargs):
        self.available_methods = ['']
        super(NotPlayingScenario, self).__init__(**kwargs)


class PlayingScenario(Draft):

    def __init__(self, **kwargs):
        self.available_methods = ['']
        super(PlayingScenario, self).__init__(**kwargs)

    def play(self):
        return True


class DraftTest(unittest.TestCase):

    def setUp(self):
        self.not_playing_scenario = NotPlayingScenario()
        self.playing_scenario = PlayingScenario()
        self.sys_argv_scenario = PlayingScenario(sys_argv=['scenario.py'])

    def test_abtraction(self):
        self.assertRaises(NotImplementedError, lambda: Draft())

    def test_too_short(self):
        self.assertRaises(ValueError, lambda: TooShortScenario())

    def test_wrong_type(self):
        self.assertRaises(TypeError, lambda: WrongTypeScenario())

    def test_not_playing(self):
        self.assertRaises(NotImplementedError, self.not_playing_scenario.play)

    def test_playing(self):
        self.assertEqual(True, self.playing_scenario.play())

    def test_sys_argv(self):
        self.assertEqual(self.sys_argv_scenario.available_methods[0], self.sys_argv_scenario.method)
        self.assertEqual(False, self.sys_argv_scenario.cheat)
        self.assertEqual(True, self.sys_argv_scenario.verbose)


class LevenshteinTest(unittest.TestCase):

    def setUp(self):
        self.s = 'GUMBO'
        self.t = 'GAMBOL'

    def test_levenshtein(self):
        self.assertEqual(2, levenshtein(self.s, self.t))


if __name__ == '__main__':
    unittest.main()
