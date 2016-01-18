"""
Extract feature vector from a layer
"""
import argparse
from time import strftime
import os
import importlib

import numpy as np


def get_current_datetime():
    return strftime('%Y%m%d_%H%M%S')


def load_model(fname):
    model = importlib.import_module('model_definitions.%s' % fname)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the image')
    parser.add_argument('--data', required=True, help='Image npy file')
    parser.add_argument('--model', required=True, help='Localization model')
    parser.add_argument('--model_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--layer', required=True, help='Name of the layer')
    args = parser.parse_args()

    print args

    timestamp = get_current_datetime()

    data_name = os.path.splitext(os.path.basename(args.data))[0]
    feats_fname = 'model_features/%s_%s_%s.npy' % (args.model, args.layer, data_name)
    print 'Will write to', feats_fname

    print 'Loading data: %s' % args.data
    X = np.memmap(args.data, mode='r', dtype=np.float32)
    X = X.reshape(-1, 3, args.size, args.size)
    print X.shape
    print

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    net = model.net
    net.initialize()
    print 'Loading model weights from %s' % model.model_fname
    net.load_params_from(model.model_fname)
    print

    print 'Extracting features from', args.layer
    feats = net.transform(X, args.layer)
    feats = feats.squeeze()
    print feats.shape
    print

    np.save(feats_fname, feats)
