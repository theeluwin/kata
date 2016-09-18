#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import numpy as np

from collections import defaultdict


class HMM:

    def __init__(self):
        self.c_tag = defaultdict(lambda: 0)
        self.c_word = defaultdict(lambda: 0)
        self.c_transition = defaultdict(lambda: defaultdict(lambda: 0))
        self.c_emission = defaultdict(lambda: defaultdict(lambda: 0))
        self.c_initial = defaultdict(lambda: 0)
        self.c_terminal = defaultdict(lambda: 0)
        self.l_transition = defaultdict(lambda: defaultdict(lambda: -np.inf))
        self.l_emission = defaultdict(lambda: defaultdict(lambda: -np.inf))
        self.l_initial = defaultdict(lambda: -np.inf)
        self.l_terminal = defaultdict(lambda: -np.inf)

    def fit(self, train_sents):
        for sent in train_sents:
            len_sent = len(sent)
            for i in range(len_sent):
                word, tag = sent[i]
                self.c_tag[tag] += 1
                self.c_word[word] += 1
                self.c_emission[tag][word] += 1
                if i > 0:
                    self.c_transition[sent[i - 1][1]][tag] += 1
                    if i == len_sent - 1:
                        self.c_terminal[tag] += 1
                else:
                    self.c_initial[tag] += 1
        self.tags = self.c_tag.keys()
        self.words = self.c_word.keys()
        for tag_from in self.tags:
            for tag_to in self.tags:
                if self.c_tag[tag_from]:
                    self.l_transition[tag_from][tag_to] = np.log(self.c_transition[tag_from][tag_to]) - np.log(self.c_tag[tag_from])
                else:
                    self.l_transition[tag_from][tag_to] = -np.inf
        c_initial_sum = sum(self.c_initial.values())
        c_terminal_sum = sum(self.c_terminal.values())
        for tag in self.tags:
            self.l_initial[tag] = np.log(self.c_initial[tag]) - np.log(c_initial_sum)
            self.l_terminal[tag] = np.log(self.c_terminal[tag]) - np.log(c_terminal_sum)
            smoothing = min([np.log(self.c_emission[tag][word]) - np.log(self.c_tag[tag]) for word in self.words if self.c_tag[tag] and self.c_emission[tag][word]])
            self.l_emission[tag] = defaultdict(lambda: smoothing)
            for word in self.words:
                if self.c_tag[tag]:
                    self.l_emission[tag][word] = np.log(self.c_emission[tag][word]) - np.log(self.c_tag[tag])
                else:
                    self.l_emission[tag][word] = smoothing

    def tag(self, words):
        len_words = len(words)
        viterbi = defaultdict(lambda: defaultdict(lambda: -np.inf))
        trace = defaultdict(lambda: defaultdict(lambda: 'bos'))
        for i in range(len_words):
            word = words[i]
            for tag in self.tags:
                if i == 0:
                    trace[tag][i] = 'bos'
                    viterbi[tag][i] = self.l_initial[tag] + self.l_emission[tag][word]
                else:
                    flow = lambda _tag: viterbi[_tag][i - 1] + self.l_transition[_tag][tag] + self.l_emission[tag][word]
                    trace[tag][i] = max(self.tags, key=flow)
                    viterbi[tag][i] = flow(trace[tag][i])
        trace['eos'][len_words] = max(self.tags, key=lambda _tag: viterbi[_tag][len_words - 1] + self.l_terminal[_tag])
        predictions = ['eos']
        for i in range(len_words, 0, -1):
            predictions.append(trace[predictions[-1]][i])
        return predictions[1:][::-1]
