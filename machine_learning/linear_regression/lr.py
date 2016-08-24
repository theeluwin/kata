#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import codecs
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


available_methods = ['gradient_descent', 'exact_solution', 'numpy_implemented']


def create_data(filepath):
    target = lambda x: 7 + 25 * x - x * x + random.random() * 10
    data = [(x, target(x)) for x in np.linspace(5, 20, 1000)]
    file = codecs.open(filepath, 'w', encoding='utf-8')
    file.write('\n'.join(['{},{}'.format(x, y) for x, y in data]) + '\n')


def read_data(filepath, cheat=False):
    data = np.array([[float(t) for t in line.split(',')] for line in codecs.open(filepath, 'r', encoding='utf-8').read().split('\n') if line])
    train_data, test_data = cross_validation.train_test_split(data, test_size=0.25)
    if cheat:
        train_data = data
    return train_data[:, 0], train_data[:, 1], test_data[:, 0], test_data[:, 1]


def draw_fitted_figure(filepath, data_x, data_y, fitted):
    space = np.linspace(data_x.min(), data_x.max(), 1000)
    plt.clf()  # todo: add label and legend
    plt.scatter(data_x, data_y, color='g', alpha=0.5)
    plt.plot(space, fitted(space), color='r')
    plt.savefig(filepath)


def evaluate(test_y, pred_y):
    print("rms: {}".format(np.sqrt(mean_squared_error(test_y, pred_y))))


def descent(train_x, train_y, learning_rate=1, verbose=False, draw_cost_figure=False):
    samples, order = train_x.shape
    order -= 1
    initializer = lambda: random.random() * 2 - 1
    bias = initializer()
    params = np.array([initializer() for i in range(1, order + 1)])
    train_x = train_x[:, 1:]
    mean_x = train_x.mean(axis=0)
    mean_y = train_y.mean()
    width_x = train_x.max(axis=0) - train_x.min(axis=0)
    width_y = train_y.max() - train_y.min()
    normalized_x = (train_x - mean_x) / width_x
    normalized_y = (train_y - mean_y) / width_y
    cost = lambda: np.square((params * normalized_x).sum(axis=1) + bias - normalized_y).sum() / 2 / samples
    update = lambda error: -learning_rate * error / samples
    epoch = 0
    costs = [cost()]
    while True:
        error_bias = ((params * normalized_x).sum(axis=1) + bias - normalized_y).sum()
        error_params = (((params * normalized_x).sum(axis=1) + bias - normalized_y)[:, np.newaxis] * normalized_x).sum(axis=0)
        bias += update(error_bias)
        params += update(error_params)
        current_cost = cost()
        if verbose and not epoch % 1000:
            print("current cost (normalized): {}".format(current_cost))
        if np.abs(costs[-1] - current_cost) < 1e-8 or epoch > 100000:  # todo: use validation set to early-stop
            break
        epoch += 1
        costs.append(current_cost)
    if draw_cost_figure:
        plt.clf()  # todo: add label and legend
        plt.plot(range(len(costs)), costs, color='b')
        plt.savefig(draw_cost_figure)
    bias = width_y * bias + mean_y - (width_y * mean_x * params / width_x).sum()
    params = width_y * params / width_x
    return np.concatenate([np.array([bias]), params])[::-1]


def exact(train_x, train_y):
    X = np.matrix(train_x)
    Y = np.matrix(train_y).T
    solution = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    return np.array(solution).flatten()[::-1]


def main(method='gradient_descent', cheat=False):
    if method not in available_methods:
        print("available methods: {}".format(', '.join(available_methods)))
        return
    train_x, train_y, test_x, test_y = read_data('data/samples.csv', cheat=cheat)
    order = 2
    if method == 'numpy_implemented':
        coeffs = np.polyfit(train_x, train_y, order)
    else:
        train_x = np.array([np.power(train_x, i) for i in range(order + 1)]).T
        if method == 'gradient_descent':
            coeffs = descent(train_x, train_y, learning_rate=1e-1, verbose=True, draw_cost_figure='output_cost.png')
        elif method == 'exact_solution':
            coeffs = exact(train_x, train_y)
    fitted = np.poly1d(coeffs)
    pred_y = fitted(test_x)
    evaluate(test_y, pred_y)
    draw_fitted_figure('output_{}.png'.format(method), test_x, test_y, fitted)


if __name__ == '__main__':
    try:
        method = sys.argv[1]
    except:
        method = 'gradient_descent'
    try:
        cheat = bool(sys.argv[2])
    except:
        cheat = False
    if method not in available_methods:
        print("usage: python {} [method]\navailable methods: {}".format(sys.argv[0], ', '.join(available_methods)))
    else:
        main(method, cheat)
