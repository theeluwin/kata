#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import codecs
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


available_methods = ['gradient_descent', 'exact_solution', 'numpy_polyfit', 'tensorflow']


def create_data():
    target = lambda x: 7 + 25 * x - x * x + random.random() * 10
    data = [(x, target(x)) for x in np.linspace(5, 20, 1000)]
    train_data, test_data = cross_validation.train_test_split(data, test_size=0.25)
    csvfy = lambda data: '\n'.join(['{},{}'.format(x, y) for x, y in data]) + '\n'
    codecs.open('data/train.csv', 'w', encoding='utf-8').write(csvfy(train_data))
    codecs.open('data/test.csv', 'w', encoding='utf-8').write(csvfy(test_data))


def read_data(filepath, cheat=False):
    lines = codecs.open(filepath, 'r', encoding='utf-8').read().split('\n')
    data = np.array([[float(t) for t in line.split(',')] for line in lines if line])
    return data[:, 0], data[:, 1]


def draw_fitted_figure(filepath, data_x, data_y, fitted):
    space = np.linspace(data_x.min(), data_x.max(), 1000)
    plt.clf()
    plt.scatter(data_x, data_y, color='g', alpha=0.5, label='samples')
    plt.plot(space, fitted(space), color='r', label='fitted')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.legend()
    plt.savefig(filepath)


def draw_cost_figure(costs, filepath):
    plt.clf()
    plt.plot(range(len(costs)), costs, color='b', label='cost')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(filepath)


def evaluate(test_y, pred_y):
    print("rms: {}".format(np.sqrt(mean_squared_error(test_y, pred_y))))


def descent(train_x, train_y, learning_rate=1e-1, verbose=False, cost_figure=False):
    samples, order = train_x.shape
    initializer = lambda: random.random() * 2 - 1
    bias = initializer()
    weights = np.array([initializer() for i in range(1, order + 1)])
    mean_x = train_x.mean(axis=0)
    mean_y = train_y.mean()
    width_x = train_x.max(axis=0) - train_x.min(axis=0)
    width_y = train_y.max() - train_y.min()
    normalized_x = (train_x - mean_x) / width_x
    normalized_y = (train_y - mean_y) / width_y
    cost = lambda: np.square((weights * normalized_x).sum(axis=1) + bias - normalized_y).sum() / 2 / samples
    update = lambda error: -learning_rate * error / samples
    epoch = 0
    costs = [cost()]
    while True:
        error_bias = ((weights * normalized_x).sum(axis=1) + bias - normalized_y).sum()
        error_weights = (((weights * normalized_x).sum(axis=1) + bias - normalized_y)[:, np.newaxis] * normalized_x).sum(axis=0)
        bias += update(error_bias)
        weights += update(error_weights)
        current_cost = cost()
        if verbose and not epoch % 1000:
            print("current cost (normalized): {}".format(current_cost))
        if np.abs(costs[-1] - current_cost) < 1e-8 or epoch > 100000:  # todo: use validation set to early-stop
            break
        epoch += 1
        costs.append(current_cost)
    if cost_figure:
        draw_cost_figure(costs, cost_figure)
    bias = width_y * bias + mean_y - (width_y * mean_x * weights / width_x).sum()
    weights = width_y * weights / width_x
    return np.concatenate([np.array([bias]), weights])[::-1]


def exact(train_x, train_y):
    X = np.matrix(train_x)
    Y = np.matrix(train_y).T
    solution = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    return np.array(solution).flatten()[::-1]


def tensor(train_x, train_y, learning_rate=1e-4, batch=100, verbose=True, cost_figure=False):
    train_x = np.matrix(train_x)
    train_y = np.matrix(train_y).T
    samples, order_x = train_x.shape
    samples, order_y = train_y.shape
    X = tf.placeholder('float', [None, order_x], name='X')
    Y = tf.placeholder('float', [None, order_y], name='Y')
    W = tf.Variable(tf.random_normal([order_x, order_y]), name='W')
    b = tf.Variable(np.random.randn(), name='b')
    pred_Y = tf.add(tf.matmul(X, W), b)
    cost = tf.reduce_sum(tf.pow(pred_Y - Y, 2)) / 2 / samples
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)
    epoch = 0
    costs = []
    while True:
        for i in range(int(samples / batch) + 1):
            session.run(optimizer, feed_dict={X: train_x[batch * i: batch * (i + 1), :], Y: train_y[batch * i: batch * (i + 1), :]})
        current_cost = session.run(cost, feed_dict={X: train_x, Y: train_y})
        if verbose and not epoch % 1000:
            print("current cost: {}".format(current_cost))
        if epoch and np.abs(costs[-1] - current_cost) < 1e-8 or epoch > 100000:  # todo: use validation set to early-stop
            break
        epoch += 1
        costs.append(current_cost)
    if cost_figure:
        draw_cost_figure(costs, cost_figure)
    bias = session.run(b)
    weights = session.run(W).flatten()
    return np.concatenate([np.array([bias]), weights])[::-1]


def main(method='gradient_descent', cheat=False):
    if method not in available_methods:
        print("available methods: {}".format(', '.join(available_methods)))
        return
    train_x, train_y = read_data('data/train.csv')
    test_x, test_y = read_data('data/test.csv')
    if cheat:
        train_x = np.concatenate([train_x, test_x])
        train_y = np.concatenate([train_y, test_y])
    order = 2
    if method == 'gradient_descent':
        train_x = np.array([np.power(train_x, i) for i in range(1, order + 1)]).T
        coeffs = descent(train_x, train_y, learning_rate=1e-1, verbose=True, cost_figure='output_cost_gradient_descent.png')
    elif method == 'exact_solution':
        train_x = np.array([np.power(train_x, i) for i in range(order + 1)]).T
        coeffs = exact(train_x, train_y)
    elif method == 'numpy_polyfit':
        coeffs = np.polyfit(train_x, train_y, order)
    elif method == 'tensorflow':
        train_x = np.array([np.power(train_x, i) for i in range(1, order + 1)]).T
        coeffs = tensor(train_x, train_y, learning_rate=1e-4, batch=100, verbose=True, cost_figure='output_cost_tensorflow.png')
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
