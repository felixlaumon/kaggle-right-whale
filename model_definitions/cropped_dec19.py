"""
ipython -i --pdb scripts/train_model.py -- --model cropped_dec19 --data 128_20151029 --use_cropped --as_grey --overwrite --no_test
"""

import numpy as np

from lasagne.layers import dnn
import lasagne as nn
import theano.tensor as T
import theano

from nolearn.lasagne import objective
from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import (
    ShuffleBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    MeanSubtractBatchiteratorMixin,
    AdjustGammaBatchIteratorMixin,
    RebalanceBatchIteratorMixin,
    make_iterator
)

from nolearn_utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping,
    StepDecay
)

from utils import TrainSplit, PushBestLoss
from utils.layers import batch_norm
from utils.nolearn_net import NeuralNet
from utils.iterators import PairBatchIteratorMixin
from utils.nonlinearities import low_temperature_softmax
from utils.layers import TiedDropoutLayer


def float32(k):
    return np.cast['float32'](k)


def conv2dbn(l, name, **kwargs):
    l = nn.layers.dnn.Conv2DDNNLayer(
        l, name=name,
        **kwargs
    )
    l = batch_norm(l, name='%sbn' % name)
    return l


model_fname = './models/cropped_dec19.pkl'
model_accuracy_fname = './models/cropped_dec19_accuracy.pkl'
model_history_fname = './models/cropped_dec19_history.pkl'
model_graph_fname = './models/cropped_dec19_history.png'

image_size = 256
batch_size = 32
n_classes = 447

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    AdjustGammaBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = dict(
    batch_size=batch_size,
    flip_horizontal_p=0.5,
    flip_vertical_p=0.5,
    affine_p=1.,
    affine_scale_choices=np.linspace(0.5, 1.5, 11),
    # affine_shear_choices=np.linspace(-0.5, 0.5, 11),
    affine_translation_choices=np.arange(-64, 64, 1),
    # affine_rotation_choices=np.arange(0, 360, 1),
    adjust_gamma_p=0.5,
    adjust_gamma_chocies=np.linspace(0.5, 1.5, 11)
)
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = dict(
    batch_size=batch_size,
)
test_iterator = TestIterator(**test_iterator_kwargs)

save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
save_training_history = SaveTrainingHistory(model_history_fname)
plot_training_history = PlotTrainingHistory(model_graph_fname)
early_stopping = EarlyStopping(patience=100)

conv_kwargs = dict(
    pad='same',
    nonlinearity=nn.nonlinearities.very_leaky_rectify
)

pool_kwargs = dict(
    pool_size=2,
)

l = nn.layers.InputLayer(name='in', shape=(None, 3, image_size, image_size))

# 256
l = conv2dbn(l, name='l1c1', num_filters=32, filter_size=(7, 7), stride=2, **conv_kwargs)
# l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='l1p', **pool_kwargs)

# 256
l = conv2dbn(l, name='l2c1', num_filters=48, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l2c2', num_filters=48, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l2c3', num_filters=48, filter_size=(3, 3), **conv_kwargs)

# 128
l = conv2dbn(l, name='l3c1', num_filters=64, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l3c2', num_filters=64, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l3c3', num_filters=64, filter_size=(3, 3), **conv_kwargs)

# 64
l = conv2dbn(l, name='l4c1', num_filters=80, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l4c2', num_filters=80, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l4c3', num_filters=80, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l4c4', num_filters=80, filter_size=(3, 3), **conv_kwargs)

# 32
l = conv2dbn(l, name='l5c1', num_filters=96, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l5c2', num_filters=96, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l5c3', num_filters=96, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l5c4', num_filters=96, filter_size=(3, 3), **conv_kwargs)

# 16
l = conv2dbn(l, name='l6c1', num_filters=128, filter_size=(3, 3), stride=2, **conv_kwargs)
l = conv2dbn(l, name='l6c2', num_filters=128, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l6c3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='l6c4', num_filters=128, filter_size=(3, 3), **conv_kwargs)

# 8
l = nn.layers.dnn.Pool2DDNNLayer(l, name='gp', pool_size=8, mode='average_inc_pad')
l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.8)

l = nn.layers.DenseLayer(l, name='out', num_units=n_classes, nonlinearity=nn.nonlinearities.softmax)


net = NeuralNet(
    layers=l,

    regression=False,
    use_label_encoder=False,

    objective_l2=1e-5,

    update=nn.updates.adam,
    update_learning_rate=theano.shared(float32(1e-3)),

    train_split=TrainSplit(0.15, random_state=42, stratify=False),
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping,
        StepDecay('update_learning_rate', start=1e-3, stop=1e-5)
    ],

    verbose=10,

    max_epochs=1500,
)
