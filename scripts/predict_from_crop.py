"""
Create submission prediction from cropped images

ipython -i --pdb scripts/predict_from_crop.py -- --data localize_oct22_128_20151108 --data_size 128 --as_grey --model cropped_nov05
"""

import argparse
import importlib
import os
import sys
from time import strftime
import numpy as np
import pandas as pd
import cPickle as pickle


def load_data(fname, size, as_grey=False):
    n = 6925

    if as_grey:
        X_fname = 'cache/X_test_cropped_grey_%s.npy' % (fname)
        X_shape = (n, 1, size, size)
    else:
        X_fname = 'cache/X_test_cropped_%s.npy' % (fname)
        X_shape = (n, 3, size, size)

    print 'Loading cropped test images from', X_fname, X_shape
    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)
    return X


def load_mean(fname, as_grey):
    if as_grey:
        mean_fname = 'cache/X_cropped_grey_%s_mean.npy' % fname
    else:
        mean_fname = 'cache/X_cropped_%s_mean.npy' % fname

    if os.path.exists(mean_fname):
        return np.load(mean_fname)
    else:
        return None


def load_model(fname):
    model = importlib.import_module('model_definitions.%s' % fname)
    return model


def load_encoder(fname='models/encoder.pkl'):
    encoder = pickle.load(open(fname, 'r'))
    return encoder


def get_current_datetime():
    return strftime('%Y%m%d_%H%M%S')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_size', required=True, type=int)
    parser.add_argument('--mean_fname')
    parser.add_argument('--as_grey', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_batch_size', default=16, help='Batch size')
    args = parser.parse_args()

    output_fname = 'submissions/%s_%s.csv' % (args.model, get_current_datetime())
    print 'Will write output to %s' % output_fname
    print

    print 'Loading sample submission'
    sample_df = pd.read_csv('data/sample_submission.csv')
    print

    print 'Loading encoder'
    encoder = load_encoder()
    classes = map(lambda x: 'whale_%05d' % x, encoder.classes_)

    print 'Loading data: %s' % args.data
    X = load_data(args.data, args.data_size, as_grey=args.as_grey)
    print

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    net = model.net
    net.initialize()
    print 'Loading model weights from %s' % model.model_fname
    net.load_params_from(model.model_fname)
    print

    if args.mean_fname is not None:
        print 'Loading mean image'
        X_mean = load_mean(args.mean_fname, args.as_grey)
        if X_mean is None:
            print 'Failed to load mean file from', args.mean_fname
            sys.exit(1)
        net.batch_iterator_train.mean = X_mean
        net.batch_iterator_test.mean = X_mean
        print 'Injected mean image'
    print

    print 'Predicting...'
    y_test_pred_proba = net.predict_proba(X)
    print

    print 'Assembling final dataframe'
    fnames = sample_df[['Image']].values
    values = np.hstack([fnames, y_test_pred_proba])
    submission_df = pd.DataFrame(values, columns=['Image'] + classes)

    print submission_df.head(1)
    print
    print len(submission_df.columns)
    print

    submission_df.to_csv(output_fname, index=False)
