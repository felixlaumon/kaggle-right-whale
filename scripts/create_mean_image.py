"""
Create mean image from cache

ipython -i --pdb scripts/create_mean_image.py -- --data 128_20151029 --use_cropped --as_grey
"""
import argparse
import os
import sys
from time import strftime
import pandas as pd
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize


def check_if_image_exists(fname):
    fname = os.path.join('data/imgs/', fname)
    return os.path.exists(fname)


def get_current_date():
    return strftime('%Y%m%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--use_cropped', action='store_true')
    parser.add_argument('--as_grey', action='store_true')
    args = parser.parse_args()

    size = int(args.data.split('_')[0])

    if args.use_cropped:
        if args.as_grey:
            X_fname = 'cache/X_cropped_grey_%s.npy' % args.data
            mean_fname = 'cache/X_cropped_grey_%s_mean.npy' % args.data
        else:
            X_fname = 'cache/X_cropped_%s.npy' % args.data
            mean_fname = 'cache/X_cropped_%s_mean.npy' % args.data

    else:
        X_fname = 'cache/X_%s.npy' % args.data
        mean_fname = 'cache/X_%s_mean.npy' % args.data

    print 'Reading images from %s' % X_fname
    print 'Will output mean to %s' % mean_fname

    n = 4543
    num_channels = 1 if args.as_grey else 3
    X_shape = (n, num_channels, size, size)
    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)
    X_mean = X.mean(axis=0)

    np.save(mean_fname, X_mean)
    print 'Done'
