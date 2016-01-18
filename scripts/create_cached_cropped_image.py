"""
Create cropped cache image to memmap

ipython -i --pdb scripts/create_cached_cropped_image.py -- --size 128 --as_grey --pad 25
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
from utils import add_padding_to_bbox


def check_if_image_exists(fname):
    fname = os.path.join('data/imgs/', fname)
    return os.path.exists(fname)


def get_current_date():
    return strftime('%Y%m%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the image')
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')
    parser.add_argument('--as_grey', action='store_true', help='Save image as grayscale')
    parser.add_argument('--pad', default=0, type=int, help='Padding of the bounding box')
    parser.add_argument('--vinh', action='store_true')

    args = parser.parse_args()

    if args.vinh:
        df = pd.read_csv('data/train_with_annotations_vinh.csv')
    else:
        df = pd.read_csv('data/train_with_annotations.csv')
    df['exist'] = df['Image'].apply(check_if_image_exists)

    print '%i does not exists' % (len(df) - df['exist'].sum())
    print df[~df['exist']]

    df = df[df['exist']]
    df = df.reset_index(drop=True)

    df['whaleID'] = df['whaleID'].apply(lambda x: x.split('_')[1])

    size_fname = '%s_pad%s' % (args.size, args.pad) if args.pad > 0 else str(args.size)

    as_grey_suffix = '_grey' if args.as_grey else ''
    vinh_suffix = '_vinh' if args.vinh else ''

    X_fname = 'cache/X_cropped%s_%s%s_%s.npy' % (as_grey_suffix, size_fname, vinh_suffix, get_current_date())
    y_fname = 'cache/y_cropped%s_%s%s_%s.npy' % (as_grey_suffix, size_fname, vinh_suffix, get_current_date())
    y_shape = (len(df))

    if args.as_grey:
        X_shape = (len(df), 1, args.size, args.size)
    else:
        X_shape = (len(df), 3, args.size, args.size)

    print X_shape, y_shape

    if os.path.exists(X_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    if os.path.exists(y_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % y_fname
        sys.exit(1)

    print 'Will write X to %s with shape of %s' % (X_fname, X_shape)
    print 'Will write y to %s with shape of %s' % (y_fname, y_shape)

    X_fp = np.memmap(X_fname, dtype=np.float32, mode='w+', shape=X_shape)
    y_fp = np.memmap(y_fname, dtype=np.int32, mode='w+', shape=y_shape)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = os.path.join('data/imgs/', row['Image'])
        whale_id = row['whaleID']
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['width'])
        h = int(row['height'])

        try:
            img = imread(fname, as_grey=args.as_grey)
            if args.pad > 0:
                x, y, w, h = add_padding_to_bbox(
                    x, y, w, h, args.pad / 100.0,
                    img.shape[1], img.shape[0]
                )
            img = img[y:y + h, x:x + w]
            img = resize(img, (args.size, args.size))

            if args.as_grey:
                img = img.reshape(-1, args.size, args.size)
            else:
                img = img.transpose(2, 0, 1)

            img = img.astype(np.float32)

            X_fp[i] = img
            y_fp[i] = whale_id

            X_fp.flush()
            y_fp.flush()
        except Exception, e:
            print '%s has failed' % i, e
