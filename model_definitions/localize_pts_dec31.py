"""
ipython -i --pdb scripts/triain_pts_model.py -- --model localize_dec31 --data 256_20151022 --overwrite
"""

import numpy as np

from lasagne.layers import dnn
import lasagne as nn

from nolearn.lasagne import NeuralNet
from utils import TrainSplit
from utils.layers import batch_norm
from utils.layer_macros import conv2dbn2 as conv2dbn

from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.iterators import (
    ShuffleBatchIteratorMixin,
    BufferedBatchIteratorMixin,
    make_iterator
)

from utils.iterators import (
    AffineTransformPtsBatchIteratorMixin
)

from nolearn_utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping
)


model_fname = './models/localize_pts_dec31.pkl'
model_history_fname = './models/localize_pts_dec31_history.pkl'
model_graph_fname = './models/localize_pts_dec31_history.png'

image_size = 256
batch_size = 16

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    BufferedBatchIteratorMixin,
    AffineTransformPtsBatchIteratorMixin
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = dict(
    batch_size=batch_size,
    buffer_size=8,
    affine_p=0.5,
    affine_transform_bbox=True,
    affine_scale_choices=np.linspace(0.75, 1.25, 21),
    affine_translation_choices=np.arange(-48, 48, 1),
    affine_rotation_choices=np.arange(-180, 180, 1)
)
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = dict(
    batch_size=batch_size,
)
test_iterator = TestIterator(**test_iterator_kwargs)

save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
save_training_history = SaveTrainingHistory(model_history_fname)
plot_training_history = PlotTrainingHistory(model_graph_fname)
early_stopping = EarlyStopping(patience=150)


conv_kwargs = dict(
    pad='same',
    nonlinearity=nn.nonlinearities.very_leaky_rectify
)

pool_kwargs = dict(
    pool_size=2,
)


# 256
l = nn.layers.InputLayer(name='in', shape=(None, 3, image_size, image_size))

l = conv2dbn(l, name='l1c1', num_filters=16, filter_size=(7, 7), **conv_kwargs)
l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='l1p', **pool_kwargs)
# 128

l = conv2dbn(l, name='l2c1', num_filters=32, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l2c2', num_filters=32, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l2c3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
# 64

l = conv2dbn(l, name='l3c1', num_filters=64, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l3c2', num_filters=64, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l3c3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
# 32

l = conv2dbn(l, name='l4c1', num_filters=128, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l4c2', num_filters=128, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l4c3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
# 16

l = conv2dbn(l, name='l5c1', num_filters=256, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l5c2', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l5c3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
# 8

l = nn.layers.dnn.Pool2DDNNLayer(l, name='gp', pool_size=8, mode='average_inc_pad')
l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.5)

l = nn.layers.DenseLayer(l, name='out', num_units=4, nonlinearity=nn.nonlinearities.identity)

net = NeuralNet(
    layers=l,

    regression=True,
    use_label_encoder=False,

    objective_loss_function=nn.objectives.squared_error,
    objective_l2=1e-7,

    update=nn.updates.adam,
    update_learning_rate=1e-4,

    train_split=TrainSplit(0.15, stratify=False, random_state=42),
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping
    ],

    verbose=10,

    max_epochs=1000,
)
