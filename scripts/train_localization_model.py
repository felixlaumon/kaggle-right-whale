import argparse
import os
import sys
import importlib

import theano
import theano.tensor as T
import numpy as np
import matplotlib
import cPickle as pickle
matplotlib.use('Agg')

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold


def load_data(fname):
    n = 4543
    size = int(fname.split('_')[0])

    X_fname = 'cache/X_%s.npy' % fname
    y_fname = 'cache/bbox_%s.npy' % fname

    X_shape = (n, 3, size, size)
    y_shape = (n, 4)

    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)
    y = np.memmap(y_fname, dtype=np.int32, mode='r', shape=y_shape)

    y = y.astype(np.float32)
    y = y / size

    return X, y


def train_test_split(X, y, test_size=0.25, random_state=42):
    n_folds = int(1 / float(test_size))
    skv = KFold(len(X), n_folds=n_folds, random_state=random_state)
    train_idx, test_idx = iter(skv).next()
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_model(fname):
    model = importlib.import_module('model_definitions.%s' % fname)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    args = parser.parse_args()

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    net = model.net
    net.initialize()

    output_exists = any([
        os.path.exists(x) for x in [model.model_fname, model.model_graph_fname, model.model_history_fname]
    ])

    if output_exists and not args.overwrite:
        print 'Model output exists. Use --overwrite'
        sys.exit(1)

    print 'Loading data: %s' % args.data
    X, y = load_data(args.data)
    print X.shape, y.shape

    if not args.no_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print X_train.shape, X_train.dtype
        print y_train.shape, y_train.dtype
        print X_test.shape, y_train.dtype
        print y_test.shape, y_test.dtype

        net.fit(X_train, y_train)

        print 'Loading best param'
        net.load_params_from(model.model_fname)
        print

        print 'Evaluating on testing set'
        y_test_pred = net.predict(X_test)
        print

        print 'Test result'
        print '==========='
        print mean_squared_error(y_test, y_test_pred)
        print

    else:
        net.fit(X, y)
