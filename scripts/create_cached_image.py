"""
Create cached images to memmap
"""
from __future__ import division
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
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')
    parser.add_argument('--vinh', action='store_true')

    args = parser.parse_args()

    if args.vinh:
        df_bbox = pd.read_csv('data/train_with_annotations_vinh.csv')
    else:
        df_bbox = pd.read_csv('data/train_with_annotations.csv')

    df_pts = pd.read_csv('data/train_with_points.csv')
    df_pts = df_pts.drop('whaleID', axis=1)

    df = df_bbox.merge(df_pts, on='Image')
    print df.columns

    df['exist'] = df['Image'].apply(check_if_image_exists)

    print '%i does not exists' % (len(df) - df['exist'].sum())
    print df[~df['exist']]

    df = df[df['exist']]
    df = df.reset_index(drop=True)

    df['whaleID'] = df['whaleID'].apply(lambda x: x.split('_')[1])

    vinh_suffix = '_vinh' if args.vinh else ''

    X_fname = 'cache/X_%s%s_%s.npy' % (args.size, vinh_suffix, get_current_date())
    y_fname = 'cache/y_%s%s_%s.npy' % (args.size, vinh_suffix, get_current_date())
    bbox_fname = 'cache/bbox_%s%s_%s.npy' % (args.size, vinh_suffix, get_current_date())
    pts_fname = 'cache/pts_%s%s_%s.npy' % (args.size, vinh_suffix, get_current_date())

    X_shape = (len(df), 3, args.size, args.size)
    y_shape = (len(df))
    bbox_shape = (len(df), 4)
    pts_shape = (len(df), 4)

    print X_shape, y_shape, bbox_shape

    if os.path.exists(X_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    if os.path.exists(y_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % y_fname
        sys.exit(1)

    if os.path.exists(bbox_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % bbox_fname
        sys.exit(1)

    if os.path.exists(pts_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % pts_fname
        sys.exit(1)

    print 'Will write X to %s with shape of %s' % (X_fname, X_shape)
    print 'Will write y to %s with shape of %s' % (y_fname, y_shape)
    print 'Will write bbox to %s with shape of %s' % (bbox_fname, bbox_shape)
    print 'Will write pts to %s with shape of %s' % (pts_fname, pts_shape)

    X_fp = np.memmap(X_fname, dtype=np.float32, mode='w+', shape=X_shape)
    y_fp = np.memmap(y_fname, dtype=np.int32, mode='w+', shape=y_shape)
    bbox_fp = np.memmap(bbox_fname, dtype=np.int32, mode='w+', shape=bbox_shape)
    pts_fp = np.memmap(pts_fname, dtype=np.int32, mode='w+', shape=pts_shape)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = os.path.join('data/imgs/', row['Image'])
        whale_id = row['whaleID']
        x = row['x']
        y = row['y']
        w = row['width']
        h = row['height']

        bonnet_tip_x = int(round(row['bonnet_tip_x']))
        bonnet_tip_y = int(round(row['bonnet_tip_y']))
        blowhead_x = int(round(row['blowhead_x']))
        blowhead_y = int(round(row['blowhead_y']))

        try:
            img = imread(fname)
            img_h, img_w, _ = img.shape
            h_scale = args.size / img_h
            w_scale = args.size / img_w
            img = resize(img, (args.size, args.size))
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)

            assert img.shape == (3, args.size, args.size)
            assert img.dtype == np.float32

            X_fp[i] = img
            y_fp[i] = whale_id

            bbox_fp[i] = [
                x * w_scale,
                y * h_scale,
                w * w_scale,
                h * h_scale
            ]

            pts_fp[i] = [
                bonnet_tip_x * w_scale, bonnet_tip_y * h_scale,
                blowhead_x * w_scale, blowhead_y * h_scale
            ]

            X_fp.flush()
            y_fp.flush()
            bbox_fp.flush()
            pts_fp.flush()
        except:
            print '%s has failed' % i
