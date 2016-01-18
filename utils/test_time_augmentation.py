import multiprocessing
import itertools
import functools
from tqdm import tqdm

import numpy as np

from nolearn_utils.image_utils import im_affine_transform
from sklearn.base import BaseEstimator


def tta_transform(x, tf_args, return_tform=False):
    z, s, r, t = tf_args
    return im_affine_transform(x, z, r, s, t, t, return_tform=return_tform)


class TestTimeAutgmentationPredictor(BaseEstimator):
    def __init__(self, net, n_jobs=2,
                 scale_choices=[1], shear_choices=[0],
                 rotation_choices=[0], translation_choices=[0]):
        self.net = net
        self.pool = multiprocessing.Pool(processes=n_jobs)
        self.scale_choices = scale_choices
        self.shear_choices = shear_choices
        self.rotation_choices = rotation_choices
        self.translation_choices = translation_choices

    def predict_proba(self, X):
        tf_args = list(itertools.product(
            self.scale_choices, self.shear_choices,
            self.rotation_choices, self.translation_choices
        ))

        print 'Number of combinations', len(tf_args)

        X_probas = []

        for x in tqdm(X, total=len(X)):
            tta_transform_x = functools.partial(tta_transform, x)
            x_tf = np.asarray(self.pool.map(tta_transform_x, tf_args))

            x_tf_pred_probas = self.net.predict_proba(x_tf)

            x_tf_pred_proba = x_tf_pred_probas.sum(axis=0)
            x_tf_pred_proba /= x_tf_pred_proba.sum()

            X_probas.append(x_tf_pred_proba)

        return np.vstack(X_probas)


class TestTimeAutgmentationPtsPredictor(TestTimeAutgmentationPredictor):
    def predict(self, X):
        image_height = X.shape[2]
        image_width = X.shape[3]
        assert image_height == image_width

        tf_args = list(itertools.product(
            self.scale_choices, self.shear_choices,
            self.rotation_choices, self.translation_choices
        ))

        print 'Number of combinations', len(tf_args)

        all_pts = []

        for x in tqdm(X, total=len(X)):
            tta_transform_x = functools.partial(tta_transform, x, return_tform=True)
            data = self.pool.map(tta_transform_x, tf_args)
            x_tf, tforms = zip(*data)
            x_tf = np.asarray(x_tf)

            pts = self.net.predict(x_tf)

            pts_tf = []
            for pt, tform in zip(pts, tforms):
                pt_tf = tform(pt.reshape(2, 2) * image_height).ravel()
                pt_tf /= image_height
                pts_tf.append(pt_tf)
            pts_tf = np.vstack(pts_tf)

            mean_pt = pts_tf.mean(axis=0)
            all_pts.append(mean_pt)

        return np.vstack(all_pts)

    def predict_proba(self, X):
        raise NotImplementedError()
