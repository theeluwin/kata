#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
import codecs
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


target = lambda x: 7 + 25 * x - x * x
available_methods = ['gradient_descent', 'exact_solution', 'numpy_polyfit', 'tensorflow']


def create_data():
    data = [(x, target(x) + random.random() * 10 - 5) for x in np.linspace(5, 20, 1000)]
    train_data, test_data = cross_validation.train_test_split(data, test_size=0.25)
    csvfy = lambda data: '\n'.join(['{},{}'.format(x, y) for x, y in data]) + '\n'
    codecs.open('data/train.csv', 'w', encoding='utf-8').write(csvfy(train_data))
    codecs.open('data/test.csv', 'w', encoding='utf-8').write(csvfy(test_data))


def read_data(filepath, cheat=False):
    lines = codecs.open(filepath, 'r', encoding='utf-8').read().split('\n')
    data = np.array([[float(t) for t in line.split(',')] for line in lines if line])
    return data[:, 0], data[:, 1]


def z_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    try:
        std[std == 0] = 1
    except:
        std = std if std else 1
    normalized = (data - mean) / std
    return normalized, mean, std


def z_rescale(weights, bias, mean_x, std_x, mean_y, std_y):
    return std_y * weights / std_x, std_y * bias + mean_y - (std_y * weights * mean_x / std_x).sum()


def plot_fit(filepath, data_x, data_y, predict, target=None):
    space = np.linspace(data_x.min(), data_x.max(), 1000)
    plt.clf()
    plt.scatter(data_x, data_y, color='g', alpha=0.5, label='samples')
    plt.plot(space, predict(space), color='r', label='predicted')
    if target:
        plt.plot(space, target(space), color='b', label='target')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.legend()
    plt.savefig(filepath)


def plot_cost(filepath, costs):
    plt.clf()
    plt.plot(range(len(costs)), costs, color='b', label='cost')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(filepath)


def evaluate(test_y, pred_y):
    print("rms: {}".format(np.sqrt(mean_squared_error(test_y, pred_y))))


def descent(train_x, train_y, learning_rate=1e-1, normalize=True, regularize=1e-4, verbose=True, cost_figure=False):
    if normalize:
        train_x, mean_x, std_x = z_normalize(train_x)
        train_y, mean_y, std_y = z_normalize(train_y)
    samples, order = train_x.shape
    initializer = lambda: random.random() * 2 - 1
    bias = initializer()
    weights = np.array([initializer() for i in range(order)])
    cost = lambda: np.square((weights * train_x).sum(axis=1) + bias - train_y).sum() / 2 / samples + regularize * (np.square(weights).sum() + bias * bias)
    update = lambda error: -learning_rate * error / samples
    epoch = 0
    costs = [cost()]
    while True:
        error_bias = ((weights * train_x).sum(axis=1) + bias - train_y).sum() + 2 * regularize * bias
        error_weights = (((weights * train_x).sum(axis=1) + bias - train_y)[:, np.newaxis] * train_x).sum(axis=0) + 2 * regularize * weights
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
        weights, bias = z_rescale(weights, bias, mean_x, std_x, mean_y, std_y)
    return np.concatenate([np.array([bias]), weights])[::-1]


def exact(train_x, train_y):
    X = np.matrix(train_x)
    Y = np.matrix(train_y).T
    solution = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    return np.array(solution).flatten()[::-1]


def tensor(train_x, train_y, learning_rate=1e-1, normalize=True, regularize=1e-4, batch=100, verbose=True, cost_figure=False):
    if normalize:
        train_x, mean_x, std_x = z_normalize(train_x)
        train_y, mean_y, std_y = z_normalize(train_y)
    train_x = np.matrix(train_x)
    train_y = np.matrix(train_y).T
    samples, order_x = train_x.shape
    samples, order_y = train_y.shape
    X = tf.placeholder('float', [None, order_x], name='X')
    Y = tf.placeholder('float', [None, order_y], name='Y')
    W = tf.Variable(tf.random_normal([order_x, order_y]), name='W')
    b = tf.Variable(np.random.randn(), name='b')
    pred_Y = tf.add(tf.matmul(X, W), b)
    if regularize > 0:
        cost = tf.reduce_sum(tf.div(tf.pow(pred_Y - Y, 2), 2 * samples)) + regularize * (tf.reduce_sum(tf.pow(W, 2)) + b * b)
    else:
        cost = tf.reduce_sum(tf.div(tf.pow(pred_Y - Y, 2), 2 * samples))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)
    epoch = 0
    costs = []
    while True:
        if batch:
            for i in range(int(samples / batch) + 1):
                session.run(optimizer, feed_dict={X: train_x[batch * i: batch * (i + 1), :], Y: train_y[batch * i: batch * (i + 1), :]})
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
    weights = session.run(W).flatten()
    bias = session.run(b)
    if normalize:
        weights, bias = z_rescale(weights, bias, mean_x, std_x, mean_y, std_y)
    return np.concatenate([np.array([bias]), weights])[::-1]


def power_chain(data, orders):
    return np.array([np.power(data, order) for order in orders]).T


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
        features = power_chain(train_x, range(1, order + 1))
        coeffs = descent(features, train_y, learning_rate=1e-1, normalize=True, regularize=1e-4, verbose=True, cost_figure='output_cost_gradient_descent.png')
    elif method == 'exact_solution':
        features = power_chain(train_x, range(order + 1))
        coeffs = exact(features, train_y)
    elif method == 'numpy_polyfit':
        coeffs = np.polyfit(train_x, train_y, order)
    elif method == 'tensorflow':
        features = power_chain(train_x, range(1, order + 1))
        coeffs = tensor(features, train_y, learning_rate=1e-1, normalize=True, regularize=1e-4, batch=100, verbose=True, cost_figure='output_cost_tensorflow.png')
    predict = np.poly1d(coeffs)
    evaluate(test_y, predict(test_x))
    plot_fit('output_{}.png'.format(method), test_x, test_y, predict, target)


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
