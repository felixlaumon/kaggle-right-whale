import argparse
import importlib
from time import strftime
import numpy as np
import pandas as pd
import cPickle as pickle


def load_data(fname):
    n = 6925
    size = int(fname.split('_')[0])
    X_fname = 'cache/X_test_%s.npy' % fname
    X_shape = (n, 3, size, size)

    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)
    return X


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
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    output_fname = 'submissions/%s.csv' % get_current_datetime()
    print 'Will write output to %s' % output_fname
    print

    print 'Loading sample submission'
    sample_df = pd.read_csv('data/sample_submission.csv')
    print

    print 'Loading encoder'
    encoder = load_encoder()
    classes = map(lambda x: 'whale_%05d' % x, encoder.classes_)

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    net = model.net
    net.initialize()
    print 'Loading model weights from %s' % model.model_fname
    net.load_weights_from(model.model_fname)
    print

    print 'Loading data: %s' % args.data
    X = load_data(args.data)
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
