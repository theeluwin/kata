#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import sys
import codecs
import pycrfsuite
import sklearn

from itertools import chain
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from hmm import HMM


def corpus2sents(filepath):
    corpus = codecs.open(filepath, 'r', encoding='utf-8').read()
    raws = corpus.split('\r\n\t\r\n')
    sents = []
    for raw in raws:
        tokens = raw.split('\r\n')
        sent = []
        for token in tokens:
            try:
                word, tag = token.split('\t')
                if word and tag:
                    sent.append([word.lower(), tag])
            except:
                pass
        sents.append(sent)
    return sents


def index2feature(sent, i, offset):
    word, tag = sent[i + offset]
    if offset < 0:
        sign = ''
    else:
        sign = '+'
    return '{}{}:word={}'.format(sign, offset, word)


def word2features(sent, i):
    L = len(sent)
    word, tag = sent[i]
    features = ['bias']
    features.append(index2feature(sent, i, 0))
    if i > 1:
        features.append(index2feature(sent, i, -2))
    if i > 0:
        features.append(index2feature(sent, i, -1))
    else:
        features.append('bos')
    if i < L - 2:
        features.append(index2feature(sent, i, 2))
    if i < L - 1:
        features.append(index2feature(sent, i, 1))
    else:
        features.append('eos')
    return features


def sent2words(sent):
    return [word for word, tag in sent]


def sent2tags(sent):
    return [tag for word, tag in sent]


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def flush_crf(filepath, X, Y):
    result = codecs.open(filepath, 'w', encoding='utf-8')
    for x, y in zip(X, Y):
        result.write(' '.join(['{}/{}'.format(feature[1].split('=')[1], tag) for feature, tag in zip(x, y)]))
        result.write('\n')
    result.close()


def flush_hmm(filepath, X, Y):
    result = codecs.open(filepath, 'w', encoding='utf-8')
    for x, y in zip(X, Y):
        result.write(' '.join(['{}/{}'.format(word, tag) for word, tag in zip(x, y)]))
        result.write('\n')
    result.close()


def evaluate(test_y, pred_y):
    total, wrong = 0, 0
    for x, y in zip(test_y, pred_y):
        for i in range(len(y)):
            total += 1
            if y[i] != x[i]:
                wrong += 1
    accuracy = int((1 - wrong / total) * 10000) / 100.0
    lb = LabelBinarizer()
    test_y_combined = lb.fit_transform(list(chain.from_iterable(test_y)))
    pred_y_combined = lb.transform(list(chain.from_iterable(pred_y)))
    tagset = sorted(set(lb.classes_))
    class_indices = {cls: idx for idx, cls in enumerate(tagset)}
    print(classification_report(test_y_combined, pred_y_combined, labels=[class_indices[cls] for cls in tagset], target_names=tagset))
    print("overall accuracy: {}".format(accuracy))


def pos_hmm():
    train_sents = corpus2sents('corpus/training.txt')
    test_sents = corpus2sents('corpus/test.txt')
    train_x = [sent2words(sent) for sent in train_sents]
    train_y = [sent2tags(sent) for sent in train_sents]
    test_x = [sent2words(sent) for sent in test_sents]
    test_y = [sent2tags(sent) for sent in test_sents]
    hmm = HMM()
    hmm.fit(train_sents)
    pred_y = [hmm.tag(x) for x in test_x]
    flush_hmm('pred_hmm.txt', test_x, pred_y)
    evaluate(test_y, pred_y)
    print("---\n")


def pos_crf():
    train_sents = corpus2sents('corpus/training.txt')
    test_sents = corpus2sents('corpus/test.txt')
    train_x = [sent2features(sent) for sent in train_sents]
    train_y = [sent2tags(sent) for sent in train_sents]
    test_x = [sent2features(sent) for sent in test_sents]
    test_y = [sent2tags(sent) for sent in test_sents]
    trainer = pycrfsuite.Trainer()
    for x, y in zip(train_x, train_y):
        trainer.append(x, y)
    trainer.train('model.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')
    pred_y = [tagger.tag(x) for x in test_x]
    flush_crf('pred_crf.txt', test_x, pred_y)
    evaluate(test_y, pred_y)
    print("---\n")


def main(method='hmm'):
    if method == 'hmm':
        pos_hmm()
    elif method == 'crf':
        pos_crf()


if __name__ == '__main__':
    available_methods = ['hmm', 'crf']
    try:
        method = sys.argv[1]
    except:
        method = 'hmm'
    if method not in available_methods:
        print("usage: python pos.py [method]\nimplemented methods: hmm, crf")
    else:
        main(method)
