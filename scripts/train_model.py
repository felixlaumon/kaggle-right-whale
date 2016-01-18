import argparse
import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu0,nvcc.fastmath=True,lib.cnmem=0.85'
import sys
import importlib
from time import strftime

import theano
import theano.tensor as T
import numpy as np
import matplotlib
import cPickle as pickle
matplotlib.use('Agg')

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def load_data(fname, use_cropped=False, as_grey=False):
    n = 4543
    size = int(fname.split('_')[0])

    if use_cropped:
        if as_grey:
            X_fname = 'cache/X_cropped_grey_%s.npy' % fname
            y_fname = 'cache/y_cropped_grey_%s.npy' % fname
        else:
            X_fname = 'cache/X_cropped_%s.npy' % fname
            y_fname = 'cache/y_cropped_%s.npy' % fname

    else:
        X_fname = 'cache/X_%s.npy' % fname
        y_fname = 'cache/y_%s.npy' % fname

    num_channels = 1 if args.as_grey else 3
    X_shape = (n, num_channels, size, size)
    y_shape = (n,)

    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)
    y = np.memmap(y_fname, dtype=np.int32, mode='r', shape=y_shape)

    assert X.shape == X_shape
    assert y.shape == y_shape

    return X, y


def load_mean(fname, use_cropped, as_grey):
    if use_cropped:
        if as_grey:
            mean_fname = 'cache/X_cropped_grey_%s_mean.npy' % fname
        else:
            mean_fname = 'cache/X_cropped_%s_mean.npy' % fname
    else:
        mean_fname = 'cache/X_%s_mean.npy' % fname

    if os.path.exists(mean_fname):
        return np.load(mean_fname)
    else:
        return None


def filter_by_min_occ(X, y, min_occ):
    occs = np.bincount(y)
    mask = np.zeros_like(y).astype(bool)

    for i, occ in enumerate(occs):
        if occ == min_occ:
            mask[y == i] = True

    return X[mask], y[mask]


def train_test_split(X, y, test_size=0.25, random_state=42, stratify=True):
    if stratify:
        n_folds = int(round(1 / test_size))
        sss = StratifiedKFold(y, n_folds=n_folds, random_state=random_state)
    else:
        sss = ShuffleSplit(len(y), test_size=test_size, random_state=random_state)
    train_idx, test_idx = iter(sss).next()
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_model(fname):
    model = importlib.import_module('model_definitions.%s' % fname)
    return model


def load_encoder(fname='models/encoder.pkl'):
    encoder = pickle.load(open(fname, 'r'))
    return encoder


def get_current_time():
    return strftime('%Y%m%d_%H%M%S')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_cropped', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--min_occ', type=int, default=None)
    parser.add_argument('--as_grey', action='store_true')
    parser.add_argument('--no_mean', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    args = parser.parse_args()

    # log_fname = 'logs/%s.log' % get_current_time()
    # print 'Will write logs to %s' % log_fname

    print 'args'
    print args
    print

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    net = model.net
    net.initialize()
    print

    output_exists = any([
        os.path.exists(x) for x in [
            model.model_fname, model.model_graph_fname, model.model_history_fname
        ]
    ])
    if output_exists and not args.overwrite:
        print 'Model output exists. Use --overwrite'
        sys.exit(1)

    print 'Loading data: %s' % args.data
    X, y = load_data(args.data, args.use_cropped, args.as_grey)
    print X.shape, y.shape
    print

    # TODO exit if the shapes don't match image_size

    if args.min_occ is not None:
        print 'Filtering dataset with min occurence of %i' % args.min_occ
        X, y = filter_by_min_occ(X, y, args.min_occ)
        print X.shape, y.shape
        print 'WARNING: update the number of units at the final layer to %i' % np.unique(y).shape[0]
        print

    print 'Loading encoder'
    encoder = load_encoder()
    # encoder = LabelEncoder().fit(y)
    y = encoder.transform(y).astype(np.int32)
    print np.unique(y).shape[0]
    print y.min(), y.max()
    print

    print 'Loading mean image'
    X_mean = load_mean(args.data, args.use_cropped, args.as_grey)
    if not args.no_mean and X_mean is not None:
        net.batch_iterator_train.mean = X_mean
        net.batch_iterator_test.mean = X_mean
        print 'Injected mean image'
    else:
        print 'Cannot load mean image'
    print

    if args.continue_training and os.path.exists(model.model_fname):
        print 'Loading model params from %s' % model.model_fname
        net.load_params_from(model.model_fname)
        with open(model.model_history_fname) as f:
            net.train_history_ = pickle.load(f)

    if not args.no_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=True)

        print 'Train / Test set split'
        print X_train.shape, X_train.dtype
        print y_train.shape, y_train.dtype
        print X_test.shape, y_train.dtype
        print y_test.shape, y_test.dtype
        print
        print 'Training set images / label: min=%i, max=%i' % (
            np.bincount(y_train).min(), np.bincount(y_train).max()
        )
        print 'Test set images / label: min=%i, max=%i' % (
            np.bincount(y_test).min(), np.bincount(y_test).max()
        )

        net.fit(X_train, y_train)

        print 'Loading best param'
        net.load_params_from(model.model_fname)
        print

        print 'Evaluating on test set'
        y_test_pred = net.predict(X_test)
        y_test_pred_proba = net.predict_proba(X_test)
        print

        print 'Classification Report'
        print '====================='
        print classification_report(y_test, y_test_pred)
        print

        print 'Accuracy Score'
        print '=============='
        score = accuracy_score(y_test, y_test_pred)
        print '%.6f' % score
        print

        print 'Logloss'
        print '======='
        logloss = log_loss(y_test, y_test_pred_proba)
        print '%.6f' % logloss
        print

    else:
        net.fit(X, y)
