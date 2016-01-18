import numpy as np

from nolearn_utils.iterators import get_random_idx
from nolearn_utils.image_utils import im_affine_transform
from scipy.ndimage import find_objects


def get_pair_idx(y, same_p=0.5):
    n = len(y)
    # labels = np.unique(y)

    left_idxes = np.zeros(n, dtype=int)
    right_idxes = np.zeros(n, dtype=int)

    left_labels = np.zeros(n, dtype=y.dtype)
    right_labels = np.zeros(n, dtype=y.dtype)

    for i in range(n):
        is_same = np.random.random() < same_p
        # Sample from the true distribution instead of the unique labels
        # so that the paired dataset have similar distribution too
        left_label = np.random.choice(y)
        if is_same:
            right_label = left_label
        else:
            right_label = np.random.choice(y)
        left_idxes[i] = np.random.choice(np.where(y == left_label)[0])
        # TODO it is possible that the left and right pair is the exact
        # same image
        right_idxes[i] = np.random.choice(np.where(y == right_label)[0])

        left_labels[i] = left_label
        right_labels[i] = right_label

    return (left_idxes, right_idxes), (left_labels, right_labels)


class RandomFlipBatchIteratorMixin(object):
    """
    Randomly flip the random horizontally or vertically
    Also flip the bounding box (y)
    """
    def __init__(self, flip_horizontal_p=0.5, flip_vertical_p=0.5, *args, **kwargs):
        super(RandomFlipBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.flip_horizontal_p = flip_horizontal_p
        self.flip_vertical_p = flip_vertical_p

    def transform(self, Xb, yb):
        Xb, yb = super(RandomFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb_flipped = Xb.copy()
        yb_flipped = yb.copy()

        if self.flip_horizontal_p > 0:
            horizontal_flip_idx = get_random_idx(Xb, self.flip_horizontal_p)
            Xb_flipped[horizontal_flip_idx] = Xb_flipped[horizontal_flip_idx, :, :, ::-1]
            yb_flipped[horizontal_flip_idx, 0] = 1 - yb_flipped[horizontal_flip_idx, 0] - yb_flipped[horizontal_flip_idx, 2]

        if self.flip_vertical_p > 0:
            vertical_flip_idx = get_random_idx(Xb, self.flip_vertical_p)
            Xb_flipped[vertical_flip_idx] = Xb_flipped[vertical_flip_idx, :, ::-1, :]
            yb_flipped[vertical_flip_idx, 1] = 1 - yb_flipped[vertical_flip_idx, 1] - yb_flipped[vertical_flip_idx, 3]

        return Xb_flipped, yb_flipped


class PairBatchIteratorMixin(object):
    def __init__(self, pair_same_p=0.5, pair_stack=True, *args, **kwargs):
        super(PairBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.pair_same_p = pair_same_p
        self.pair_stack = pair_stack

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        n_batches = (n_samples + bs - 1) // bs

        (left_idxes, right_idxes), (left_labels, right_labels) = get_pair_idx(self.y, same_p=self.pair_same_p)

        for i in range(n_batches):
            sl = slice(i * bs, (i + 1) * bs)
            Xb_left = self.X[left_idxes[sl]]
            Xb_right = self.X[right_idxes[sl]]

            if self.y is not None:
                yb_left = self.y[left_idxes[sl]]
                yb_right = self.y[right_idxes[sl]]
            else:
                yb_left = None
                yb_right = None

            Xb_left, yb_left = self.transform(Xb_left, yb_left)
            Xb_right, yb_right = self.transform(Xb_right, yb_right)

            if self.pair_stack == 'hstack':
                yield np.hstack([Xb_left, Xb_right]), np.vstack([yb_left, yb_right]).T
            elif self.pair_stack == 'oddeven':
                yield np.hstack([Xb_left, Xb_right]).reshape(-1, Xb_left.shape[1], Xb_left.shape[2], Xb_left.shape[3]), np.vstack([yb_left, yb_right]).T.ravel()
            else:
                yield (Xb_left, Xb_right), (yb_left, yb_right)


class AffineTransformBBoxBatchIteratorMixin(object):
    """
    Apply affine transform (scale, translate and rotation)
    with a random chance
    """
    def __init__(self, affine_p,
                 affine_scale_choices=[1.], affine_translation_choices=[0.],
                 affine_rotation_choices=[0.], affine_shear_choices=[0.],
                 affine_transform_bbox=False,
                 *args, **kwargs):
        super(AffineTransformBBoxBatchIteratorMixin,
              self).__init__(*args, **kwargs)
        self.affine_p = affine_p
        self.affine_scale_choices = affine_scale_choices
        self.affine_translation_choices = affine_translation_choices
        self.affine_rotation_choices = affine_rotation_choices
        self.affine_shear_choices = affine_shear_choices

        if self.verbose:
            print('Random transform probability: %.2f' % self.affine_p)
            print('Rotation choices', self.affine_rotation_choices)
            print('Scale choices', self.affine_scale_choices)
            print('Translation choices', self.affine_translation_choices)
            print('Shear choices', self.affine_shear_choices)

    def transform(self, Xb, yb):
        Xb, yb = super(AffineTransformBBoxBatchIteratorMixin,
                       self).transform(Xb, yb)
        # Skip if affine_p is 0. Setting affine_p may be useful for quickly
        # disabling affine transformation
        if self.affine_p == 0:
            return Xb, yb

        image_height = Xb.shape[2]
        image_width = Xb.shape[3]

        assert image_height == image_width

        idx = get_random_idx(Xb, self.affine_p)
        Xb_transformed = Xb.copy()
        yb_transformed = yb.copy()

        for i in idx:
            scale = np.random.choice(self.affine_scale_choices)
            rotation = np.random.choice(self.affine_rotation_choices)
            shear = np.random.choice(self.affine_shear_choices)
            translation_y = np.random.choice(self.affine_translation_choices)
            translation_x = np.random.choice(self.affine_translation_choices)
            transform_kwargs = dict(
                scale=scale, rotation=rotation,
                shear=shear,
                translation_y=translation_y,
                translation_x=translation_x
            )

            img_transformed = im_affine_transform(
                Xb[i], **transform_kwargs)
            bbox_transformed = get_transformed_bbox(
                yb[i] * image_width, image_width, image_height, **transform_kwargs)

            Xb_transformed[i] = img_transformed
            yb_transformed[i] = np.array(bbox_transformed).astype(np.float32) / image_width

        return Xb_transformed, yb_transformed


def get_transformed_bbox(bbox, image_width, image_height, **kwargs):
    l, t, w, h = bbox
    r = l + w
    b = t + h
    y_heatmap = np.zeros((image_height, image_width)).astype(bool)
    y_heatmap[t:b, l:r] = True

    y_heatmap = im_affine_transform(y_heatmap[np.newaxis, ...], **kwargs)
    y_heatmap = y_heatmap[0].astype(bool)

    dets = find_objects(y_heatmap)

    if len(dets) == 1:
        t = dets[0][0].start
        b = dets[0][0].stop
        l = dets[0][1].start
        r = dets[0][1].stop
        w = r - l
        h = b - t
    else:
        l, t, w, h = 0, 0, 0, 0

    return l, t, w, h


class AffineTransformPtsBatchIteratorMixin(object):
    """
    Apply affine transform (scale, translate and rotation)
    with a random chance
    """
    def __init__(self, affine_p,
                 affine_scale_choices=[1.], affine_translation_choices=[0.],
                 affine_rotation_choices=[0.], affine_shear_choices=[0.],
                 affine_transform_bbox=False,
                 *args, **kwargs):
        super(AffineTransformPtsBatchIteratorMixin,
              self).__init__(*args, **kwargs)
        self.affine_p = affine_p
        self.affine_scale_choices = affine_scale_choices
        self.affine_translation_choices = affine_translation_choices
        self.affine_rotation_choices = affine_rotation_choices
        self.affine_shear_choices = affine_shear_choices

        if self.verbose:
            print('Random transform probability: %.2f' % self.affine_p)
            print('Rotation choices', self.affine_rotation_choices)
            print('Scale choices', self.affine_scale_choices)
            print('Translation choices', self.affine_translation_choices)
            print('Shear choices', self.affine_shear_choices)

    def transform(self, Xb, yb):
        Xb, yb = super(AffineTransformPtsBatchIteratorMixin,
                       self).transform(Xb, yb)
        # Skip if affine_p is 0. Setting affine_p may be useful for quickly
        # disabling affine transformation
        if self.affine_p == 0:
            return Xb, yb

        image_height = Xb.shape[2]
        image_width = Xb.shape[3]

        assert image_height == image_width

        idx = get_random_idx(Xb, self.affine_p)
        Xb_transformed = Xb.copy()
        yb_transformed = yb.copy()

        for i in idx:
            scale = np.random.choice(self.affine_scale_choices)
            rotation = np.random.choice(self.affine_rotation_choices)
            shear = np.random.choice(self.affine_shear_choices)
            translation_y = np.random.choice(self.affine_translation_choices)
            translation_x = np.random.choice(self.affine_translation_choices)

            transform_kwargs = dict(
                scale=scale, rotation=rotation,
                shear=shear,
                translation_y=translation_y,
                translation_x=translation_x,
                return_tform=True
            )

            img_transformed, tform = im_affine_transform(
                Xb[i], **transform_kwargs
            )

            Xb_transformed[i] = img_transformed

            pts = yb_transformed[i].reshape(-1, 2) * image_height
            pts = tform.inverse(pts).ravel()
            yb_transformed[i] = pts / image_height

        return Xb_transformed, yb_transformed
