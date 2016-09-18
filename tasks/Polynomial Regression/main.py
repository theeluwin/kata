#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import sys
import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


target = lambda x: 7 + 25 * x - x * x
available_methods = ['gradient_descent', 'exact_solution', 'numpy_polyfit', 'tensorflow']


def create_data():
    data = [(x, target(x) + np.random.randn() * 3) for x in np.linspace(5, 20, 1000)]
    train_data, test_data = data[:750], data[750:]
    csvfy = lambda data: '\n'.join(['{},{}'.format(x, y) for x, y in data]) + '\n'
    codecs.open('data/train.csv', 'w', encoding='utf-8').write(csvfy(train_data))
    codecs.open('data/test.csv', 'w', encoding='utf-8').write(csvfy(test_data))


def read_data(filepath, cheat=False):
    lines = codecs.open(filepath, 'r', encoding='utf-8').read().split('\n')
    data = np.asarray([[float(t) for t in line.split(',')] for line in lines if line])
    return np.split(data, [-1], axis=1)


def z_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    try:
        std[std == 0] = 1
    except:
        std = std if std else 1
    normalized = (data - mean) / std
    return normalized, mean, std


def z_rescale(bias, weights, mean_x, std_x, mean_y, std_y):
    amplifier = std_y * weights.T / std_x
    return std_y * bias + mean_y - (amplifier * mean_x).sum(), amplifier.T


def plot_fit(filepath, train_x, train_y, test_x, test_y, predict, target=None):
    space = np.linspace(min(train_x.min(), test_x.min()), max(train_x.max(), test_x.max()), 1000)
    plt.clf()
    subplot = plt.subplot()
    subplot.scatter(train_x, train_y, color='g', alpha=0.5, label='train samples')
    subplot.scatter(test_x, test_y, color='c', alpha=0.5, label='test samples')
    subplot.plot(space, predict(space), color='r', label='predicted')
    if target:
        subplot.plot(space, target(space), color='b', label='target')
    box = subplot.get_position()
    subplot.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    subplot.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.savefig(filepath)


def plot_cost(filepath, costs):
    plt.clf()
    subplot = plt.subplot()
    subplot.plot(range(len(costs)), costs, color='b', label='cost')
    subplot.legend()
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.savefig(filepath)


def evaluate(test_y, pred_y, cheated=False):
    print("rms{}: {}".format(" (cheated)" if cheated else '', np.sqrt(mean_squared_error(test_y, pred_y))))


def descent(train_x, train_y, learning_rate=1e-1, normalize=True, regularize=1e-2, verbose=True, cost_figure=False):
    if normalize:
        train_x, mean_x, std_x = z_normalize(train_x)
        train_y, mean_y, std_y = z_normalize(train_y)
    samples, order_x = train_x.shape
    samples, order_y = train_y.shape
    bias = np.random.randn()
    weights = np.random.randn(order_x, order_y)
    cost = lambda: np.asscalar((np.square(train_x.dot(weights) + bias - train_y).sum() + regularize * weights.T.dot(weights)) / (2 * samples))
    update = lambda error: -learning_rate * error / samples
    epoch = 0
    costs = [cost()]
    while True:
        error_bias = (train_x.dot(weights) + bias - train_y).sum()
        error_weights = train_x.T.dot(train_x.dot(weights) + bias - train_y) + regularize * weights
        bias += update(error_bias)
        weights += update(error_weights)
        current_cost = cost()
        if verbose and not epoch % 1000:
            print("current cost{}: {}".format(" (normalized)" if normalize else "", current_cost))
        if np.abs(costs[-1] - current_cost) < 1e-6 or epoch > 10000:  # todo: use validation set to early-stop
            break
        epoch += 1
        costs.append(current_cost)
    if cost_figure:
        plot_cost(cost_figure, costs)
    if normalize:
        bias, weights = z_rescale(bias, weights, mean_x, std_x, mean_y, std_y)
    return np.flipud(np.concatenate([bias, weights.flatten()]))


def exact(train_x, train_y, regularize=1e-2):
    square = train_x.T.dot(train_x)
    n, n = square.shape
    diagonal = np.eye(n)
    diagonal[0, 0] = 0
    solution = np.linalg.solve(square + regularize * diagonal, train_x.T.dot(train_y))
    return np.flipud(solution.flatten())


def tensor(train_x, train_y, learning_rate=1e-1, normalize=True, regularize=1e-2, batch=100, verbose=True, cost_figure=False):
    if normalize:
        train_x, mean_x, std_x = z_normalize(train_x)
        train_y, mean_y, std_y = z_normalize(train_y)
    samples, order_x = train_x.shape
    samples, order_y = train_y.shape
    X = tf.placeholder('float', [None, order_x], name='X')
    Y = tf.placeholder('float', [None, order_y], name='Y')
    b = tf.Variable(tf.random_normal(()), name='b')
    W = tf.Variable(tf.random_normal([order_x, order_y]), name='W')
    pred_Y = tf.add(tf.matmul(X, W), b)
    cost = tf.div(tf.reduce_sum(tf.square(pred_Y - Y)) + regularize * tf.reduce_sum(tf.square(W)), 2 * samples)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)
    epoch = 0
    costs = []
    while True:
        data = np.concatenate([train_x, train_y], axis=1)
        np.random.shuffle(data)
        train_x, train_y = np.split(data, [-order_y], axis=1)
        if batch:
            for i in range(int(samples / batch) + 1):
                session.run(optimizer, feed_dict={
                    X: train_x[batch * i: batch * (i + 1), :],
                    Y: train_y[batch * i: batch * (i + 1), :],
                })
        else:
            session.run(optimizer, feed_dict={X: train_x, Y: train_y})
        current_cost = session.run(cost, feed_dict={X: train_x, Y: train_y})
        if verbose and not epoch % 1000:
            print("current cost{}: {}".format(" (normalized)" if normalize else "", current_cost))
        if epoch and np.abs(costs[-1] - current_cost) < 1e-6 or epoch > 10000:  # todo: use validation set to early-stop
            break
        epoch += 1
        costs.append(current_cost)
    if cost_figure:
        plot_cost(cost_figure, costs)
    bias = session.run(b)
    weights = session.run(W)
    if normalize:
        bias, weights = z_rescale(bias, weights, mean_x, std_x, mean_y, std_y)
    return np.flipud(np.concatenate([bias, weights.flatten()]))


def power_chain(data, degrees):
    return np.concatenate([np.power(data, degree) for degree in degrees], axis=1)


def main(method='gradient_descent', cheat=False):
    if method not in available_methods:
        print("available methods: {}".format(', '.join(available_methods)))
        return
    train_x, train_y = read_data('data/train.csv')
    test_x, test_y = read_data('data/test.csv')
    if cheat:
        train_x = np.concatenate([train_x, test_x], axis=0)
        train_y = np.concatenate([train_y, test_y], axis=0)
    fit_figure = 'output_fit_{}{}.png'.format(method, "_cheated" if cheat else '')
    cost_figure = 'output_cost_{}{}.png'.format(method, "_cheated" if cheat else '')
    degree = 2
    if method == 'gradient_descent':
        features = power_chain(train_x, range(1, degree + 1))
        coeffs = descent(features, train_y, learning_rate=1e-1, normalize=True, regularize=1e-2, verbose=True, cost_figure=cost_figure)
    elif method == 'exact_solution':
        features = power_chain(train_x, range(degree + 1))
        coeffs = exact(features, train_y, regularize=1e-2)
    elif method == 'numpy_polyfit':
        coeffs = np.polyfit(train_x.flatten(), train_y.flatten(), degree)
    elif method == 'tensorflow':
        features = power_chain(train_x, range(1, degree + 1))
        coeffs = tensor(features, train_y, learning_rate=1e-1, normalize=True, regularize=1e-2, batch=100, verbose=True, cost_figure=cost_figure)
    predict = np.poly1d(coeffs)
    evaluate(test_y, predict(test_x), cheated=cheat)
    plot_fit(fit_figure, train_x, train_y, test_x, test_y, predict, target)


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
