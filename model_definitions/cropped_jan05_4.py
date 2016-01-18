"""
ipython -i --pdb scripts/train_model.py -- --model cropped_jan05_4 --data 128_20151029 --use_cropped --as_grey --overwrite --no_test
"""
import sys
sys.setrecursionlimit(99999)

import numpy as np


from lasagne.layers import dnn
import lasagne as nn
import theano.tensor as T
import theano

from utils.nolearn_net import NeuralNet
from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import (
    ShuffleBatchIteratorMixin,
    BufferedBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    AdjustGammaBatchIteratorMixin,
    make_iterator
)

from nolearn_utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping,
    StepDecay
)

from utils import TrainSplit
# from utils.layers import batch_norm
# from utils.iterators import PairBatchIteratorMixin
# from utils.nonlinearities import low_temperature_softmax
# from utils.layers import TiedDropoutLayer
from utils.layer_macros import conv2dbn
from utils.layer_macros import residual_block3_localbn as residual_block


def float32(k):
    return np.cast['float32'](k)


model_fname = './models/cropped_jan05_4.pkl'
model_accuracy_fname = './models/cropped_jan05_4_accuracy.pkl'
model_history_fname = './models/cropped_jan05_4_history.pkl'
model_graph_fname = './models/cropped_jan05_4_history.png'

image_size = 256
batch_size = 16
n_classes = 447

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    BufferedBatchIteratorMixin,
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
    buffer_size=16,
    flip_horizontal_p=0.5,
    flip_vertical_p=0.5,
    affine_p=1.,
    affine_scale_choices=np.linspace(0.5, 1.5, 11),
    affine_shear_choices=np.linspace(-0.25, 0.25, 11),
    affine_translation_choices=np.arange(-32, 32, 1),
    affine_rotation_choices=np.arange(-45, 45, 1),
    adjust_gamma_p=0.5,
    adjust_gamma_chocies=np.linspace(0.8, 1.2, 11)
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
    nonlinearity=nn.nonlinearities.very_leaky_rectify,
)

l = nn.layers.InputLayer(name='in', shape=(None, 3, image_size, image_size))
# 256x256

l = conv2dbn(
    l, name='l1c1', num_filters=32, filter_size=(7, 7), stride=2,
    **conv_kwargs
)
# 128x128

l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='l1p', pool_size=(3, 3), stride=2)
# 64x64

for i in range(3):
    l = residual_block(
        l, name='2c%s' % i,
        # bottleneck=True, bottleneck_factor=4,
        num_filters=48, filter_size=(3, 3),
        num_layers=2,
        **conv_kwargs
    )
# 64x64

for i in range(4):
    actual_stride = 2 if i == 0 else 1
    l = residual_block(
        l, name='3c%s' % i,
        # bottleneck=True, bottleneck_factor=4,
        num_filters=64, filter_size=(3, 3), stride=actual_stride,
        num_layers=2,
        **conv_kwargs
    )
# 32x32

for i in range(23):
    actual_stride = 2 if i == 0 else 1
    l = residual_block(
        l, name='4c%s' % i,
        # bottleneck=True, bottleneck_factor=4,
        num_filters=80, filter_size=(3, 3), stride=actual_stride,
        num_layers=2,
        **conv_kwargs
    )
# 16x16

for i in range(3):
    actual_stride = 2 if i == 0 else 1
    l = residual_block(
        l, name='5c%s' % i,
        # bottleneck=True, bottleneck_factor=4,
        num_filters=128, filter_size=(3, 3), stride=actual_stride,
        num_layers=2,
        **conv_kwargs
    )
# 8x8

# 8
l = nn.layers.dnn.Pool2DDNNLayer(l, name='gp', pool_size=8, mode='average_inc_pad')
l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.8)

l = nn.layers.DenseLayer(l, name='out', num_units=n_classes, nonlinearity=nn.nonlinearities.softmax)


net = NeuralNet(
    layers=l,

    regression=False,
    use_label_encoder=False,

    objective_l2=1e-6,

    update=nn.updates.adam,
    update_learning_rate=1e-3,

    # update=nn.updates.nesterov_momentum,
    # update_learning_rate=theano.shared(float32(1e-1)),

    train_split=TrainSplit(0.15, random_state=42, stratify=False),
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping,
        # StepDecay('update_learning_rate', start=1e-1, stop=1e-5)
    ],

    verbose=10,

    max_epochs=2000,
)
