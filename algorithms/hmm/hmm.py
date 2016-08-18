#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

from collections import defaultdict


class HMM:

    def __init__(self):
        self.c_tag = defaultdict(lambda: 0)
        self.c_word = defaultdict(lambda: 0)
        self.c_transition = defaultdict(lambda: defaultdict(lambda: 0))
        self.c_emission = defaultdict(lambda: defaultdict(lambda: 0))
        self.c_initial = defaultdict(lambda: 0)
        self.c_terminal = defaultdict(lambda: 0)
        self.p_transition = defaultdict(lambda: defaultdict(lambda: 0))
        self.p_emission = defaultdict(lambda: defaultdict(lambda: 0))
        self.p_initial = defaultdict(lambda: 0)
        self.p_terminal = defaultdict(lambda: 0)

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
                    self.p_transition[tag_from][tag_to] = self.c_transition[tag_from][tag_to] / self.c_tag[tag_from]
                else:
                    self.p_transition[tag_from][tag_to] = 0
        c_initial_sum = sum(self.c_initial.values())
        c_terminal_sum = sum(self.c_terminal.values())
        for tag in self.tags:
            self.p_initial[tag] = self.c_initial[tag] / c_initial_sum
            self.p_terminal[tag] = self.c_terminal[tag] / c_terminal_sum
            emissions = []
            for word in self.words:
                if self.c_tag[tag] and self.c_emission[tag][word]:
                    emissions.append(self.c_emission[tag][word] / self.c_tag[tag])
            smoothing = min(emissions)
            self.p_emission[tag] = defaultdict(lambda: smoothing)
            for word in self.words:
                if self.c_tag[tag]:
                    self.p_emission[tag][word] = self.c_emission[tag][word] / self.c_tag[tag]
                else:
                    self.p_emission[tag][word] = smoothing

    def tag(self, words):
        len_words = len(words)
        argmax = lambda function, iterable: max(iterable, key=function)
        viterbi = defaultdict(lambda: defaultdict(lambda: 0))
        trace = defaultdict(lambda: defaultdict(lambda: 'bos'))
        if len_words == 0:
            return []
        for i in range(len_words):
            word = words[i]
            for tag in self.tags:
                if i == 0:
                    trace[tag][i] = 'bos'
                    viterbi[tag][i] = self.p_initial[tag] * self.p_emission[tag][word]
                else:
                    flow = lambda _tag: viterbi[_tag][i - 1] * self.p_transition[_tag][tag]
                    trace[tag][i] = argmax(flow, self.tags)
                    viterbi[tag][i] = flow(trace[tag][i]) * self.p_emission[tag][word]
        trace['eos'][len_words] = argmax(lambda _tag: viterbi[_tag][len_words - 1] * self.p_terminal[_tag], self.tags)
        predictions = ['eos']
        for i in range(len_words, 0, -1):
            predictions.append(trace[predictions[-1]][i])
        return predictions[1:][::-1]
