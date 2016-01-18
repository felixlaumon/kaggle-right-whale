"""
Create head cropped test set image
"""
import argparse
import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu1,nvcc.fastmath=True'
import sys
from time import strftime
import pandas as pd
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize
from utils import add_padding_to_bbox
from utils.points_crop import get_head_crop
from utils.test_time_augmentation import TestTimeAutgmentationPtsPredictor
import importlib


def get_cropped_test_img(fname, bbox_pred, pad=None, as_grey=False, return_bbox=False):
    img = imread(fname, as_grey=as_grey)
    h = img.shape[0]
    w = img.shape[1]
    bbox_pred = bbox_pred * [w, h, w, h]
    bbox_pred = np.round(bbox_pred).astype(int)
    l = min(max(bbox_pred[0], 0), w)
    t = min(max(bbox_pred[1], 0), h)
    r = min(max(l + bbox_pred[2], 0), w)
    b = min(max(t + bbox_pred[3], 0), h)

    if pad is not None:
        l, t, r, b = add_padding_to_bbox(
            l, t, (r - l), (b - t), pad / 100.0,
            img.shape[1], img.shape[0],
            format='ltrb'
        )
    cropped_img = img[t:b, l:r]

    if return_bbox:
        return cropped_img, bbox_pred
    else:
        return cropped_img


def load_data(fname, data_grey=False):
    n = 6925
    size = int(fname.split('_')[0])

    if data_grey:
        X_fname = 'cache/X_test_grey_%s.npy' % fname
    else:
        X_fname = 'cache/X_test_%s.npy' % fname

    num_channels = 1 if data_grey else 3
    X_shape = (n, num_channels, size, size)

    print 'Load test data from %s' % X_fname
    X = np.memmap(X_fname, dtype=np.float32, mode='r', shape=X_shape)

    return X


def get_current_date():
    return strftime('%Y%m%d')


def load_model(fname):
    model = importlib.import_module('model_definitions.%s' % fname)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the image')
    parser.add_argument('--data', required=True, help='Input uncropped image')
    parser.add_argument('--data_grey', action='store_true', help='Is the data grey?')
    parser.add_argument('--model', required=True, help='Localization model')
    parser.add_argument('--model_batch_size', default=16, help='Batch size')
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')
    parser.add_argument('--as_grey', action='store_true', help='Save image as grayscale')
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()

    print 'Loading model: %s' % args.model
    model = load_model(args.model)
    localization_net = model.net
    model_fname = model.model_fname[2:]  # Hack to remove the "./"
    localization_net.load_params_from(model_fname)
    localization_net.batch_iterator_train.batch_size = args.model_batch_size
    localization_net.batch_iterator_test.batch_size = args.model_batch_size
    print

    print 'Loading data: %s' % args.data
    df = pd.read_csv('data/sample_submission.csv')
    X_test = load_data(args.data, args.data_grey)
    print X_test.shape
    print

    print 'Preparing output'
    size_fname = str(args.size)
    tta_suffix = 'tta_' if args.tta else ''
    if args.as_grey:
        X_fname = 'cache/X_test_head_crop_grey_%s_%s_%s%s.npy' % (
            args.model, size_fname, tta_suffix, get_current_date()
        )
        X_shape = (len(df), 1, args.size, args.size)
    else:
        X_fname = 'cache/X_test_head_crop_%s_%s_%s%s.npy' % (
            args.model, size_fname, tta_suffix, get_current_date()
        )
        X_shape = (len(df), 3, args.size, args.size)

    if os.path.exists(X_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    print 'Will write X_test_cropped to %s with shape of %s' % (X_fname, X_shape)
    print

    X_fp = np.memmap(X_fname, dtype=np.float32, mode='w+', shape=X_shape)

    print 'Predicting points'
    if args.tta:
        print 'Using test time augmentation'
        scale_choices = [0.75, 1, 1.25]
        shear_choices = [-0.25, 0, 0.25]
        rotation_choices = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        translation_choices = [-25, 0, 25]
        tta = TestTimeAutgmentationPtsPredictor(
            localization_net, n_jobs=12,
            scale_choices=scale_choices,
            shear_choices=shear_choices,
            rotation_choices=rotation_choices,
            translation_choices=translation_choices
        )

        test_pts_pred = tta.predict(X_test)
    else:
        test_pts_pred = localization_net.predict(X_test)
    print

    assert len(test_pts_pred) == len(X_fp)

    for (i, row), pts in tqdm(zip(df.iterrows(), test_pts_pred), total=len(df)):
        fname = os.path.join('data/imgs/', row['Image'])
        pts = pts.reshape(2, 2)
        pt1 = pts[0]
        pt2 = pts[1]

        try:
            img = imread(fname, as_grey=args.as_grey)
            h = img.shape[0]
            w = img.shape[1]
            pt1 = pt1 * [w, h]
            pt2 = pt2 * [w, h]
            cropped_img = get_head_crop(img, pt1, pt2)
            cropped_img = resize(cropped_img, (args.size, args.size))
            cropped_img = cropped_img.astype(np.float32)

            if args.as_grey:
                cropped_img = cropped_img.reshape(1, args.size, args.size)
            else:
                cropped_img = cropped_img.transpose(2, 0, 1)

            assert cropped_img.dtype == np.float32

            X_fp[i] = cropped_img
            X_fp.flush()
        except Exception, e:
            print '%s has failed' % i
            print e
            print pts
