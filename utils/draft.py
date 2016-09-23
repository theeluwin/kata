# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import time

from fabric import colors


class Draft(object):

    def __init__(self, **kwargs):
        try:
            if not len(self.available_methods):
                raise ValueError("'available_methods' should have at least one element")
        except AttributeError:
            raise NotImplementedError("this is an abstract class; declare property 'available_methods'")
        except TypeError:
            raise TypeError("'available_methods' should be a len-able")
        self.method = str(kwargs.get('method', self.available_methods[0]))
        self.cheat = bool(kwargs.get('cheat', False))
        self.verbose = bool(kwargs.get('verbose', True))
        if 'sys_argv' in kwargs:
            arguments = kwargs['sys_argv']
            self.method = str(arguments[1]) if len(arguments) > 1 else self.available_methods[0]
            self.cheat = bool(arguments[2]) if len(arguments) > 2 else False
            self.verbose = bool(arguments[3]) if len(arguments) > 3 else True
            if self.method not in self.available_methods and self.verbose:
                print("usage: python {} [method]".format(arguments[0]))
        if self.method not in self.available_methods:
            raise IndexError("available methods: {}".format(', '.join(self.available_methods)))

    @staticmethod
    def verbose_elapsed(interval):
        if interval < 60:
            return "%4.02fs" % interval
        interval = int(interval)
        days = int(interval / 86400)
        hours = int(interval / 3600) % 24
        minutes = int(interval / 60) % 60
        seconds = interval % 60
        verbose = ""
        verbose += "{}d ".format(days) if days else ""
        verbose += "{}h ".format(hours) if hours else ""
        verbose += "{}m ".format(minutes) if minutes else ""
        verbose += "{}s".format(seconds)
        return verbose

    def print_elapsed(f):
        def wrapper(self, *args, **kwargs):
            stamp = time.time()
            result = f(self, *args, **kwargs)
            if self.verbose:
                print(colors.green("elapsed time of '{}': {}".format(f.__name__, Draft.verbose_elapsed(time.time() - stamp))))
            return result
        return wrapper

    def play(self):
        raise NotImplementedError()

    print_elapsed = staticmethod(print_elapsed)
