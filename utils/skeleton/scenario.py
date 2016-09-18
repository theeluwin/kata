#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys

from os.path import dirname, abspath

sys.path.append((lambda f, x: f(f(f(x))))(dirname, abspath(__file__)))

from utils.draft import Draft

#------------------------------------------------------------------------------#

import time


class Scenario(Draft):

    def __init__(self, **kwargs):
        self.available_methods = ['baseline']
        super(Scenario, self).__init__(**kwargs)

    def baseline(self):
        time.sleep(1)
        if self.verbose:
            print("Hello, world!")
        return 1

    @Draft.print_elapsed
    def play(self):
        result = self.baseline()
        return result

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    scenario = Scenario(sys_argv=sys.argv)
    scenario.play()
