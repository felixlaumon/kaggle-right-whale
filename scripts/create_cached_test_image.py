"""
Create cached images to memmap
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


def load_images(df):
    for i, row in df['Image'].iterrows():
        img = imread(row)
        yield img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the image')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache')

    args = parser.parse_args()

    df = pd.read_csv('data/sample_submission.csv')
    df['exist'] = df['Image'].apply(check_if_image_exists)

    print '%i does not exists' % (len(df) - df['exist'].sum())

    df = df[df['exist']]
    df = df.reset_index(drop=True)

    X_fname = 'cache/X_test_%s_%s.npy' % (args.size, get_current_date())
    X_shape = (len(df), 3, args.size, args.size)

    print X_shape

    if os.path.exists(X_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    print 'Will write X to %s with shape of %s' % (X_fname, X_shape)

    X_fp = np.memmap(X_fname, dtype=np.float32, mode='w+', shape=X_shape)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = os.path.join('data/imgs/', row['Image'])

        try:
            img = imread(fname)
            img = resize(img, (args.size, args.size))
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)

            assert img.shape == (3, args.size, args.size)
            assert img.dtype == np.float32

            X_fp[i] = img
            X_fp.flush()
        except:
            print '%s has failed' % i
