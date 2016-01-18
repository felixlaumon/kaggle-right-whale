"""
ipython -i --pdb scripts/train_model.py -- --model cropped_jan05_3 --data 128_20151029 --use_cropped --as_grey --overwrite --no_test
"""

import numpy as np

from lasagne.layers import dnn
import lasagne as nn
import theano.tensor as T
import theano

from nolearn.lasagne.handlers import SaveWeights

from nolearn_utils.iterators import (
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    AdjustGammaBatchIteratorMixin,
    BufferedBatchIteratorMixin,
    make_iterator
)

from nolearn_utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping,
    StepDecay
)

from utils import TrainSplit
from utils.nolearn_net import NeuralNet
from utils.layer_macros import conv2dbn
from utils.inception import inceptionA, inceptionB, inceptionC, inceptionD, inceptionE


def float32(k):
    return np.cast['float32'](k)


model_fname = './models/cropped_jan05_3.pkl'
model_accuracy_fname = './models/cropped_jan05_3_accuracy.pkl'
model_history_fname = './models/cropped_jan05_3_history.pkl'
model_graph_fname = './models/cropped_jan05_3_history.png'

image_size = 299
batch_size = 16
n_classes = 447

train_iterator_mixins = [
    # ShuffleBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    AdjustGammaBatchIteratorMixin,
    BufferedBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = dict(
    batch_size=batch_size,
    buffer_size=32,
    shuffle=True,
    flip_horizontal_p=0.5,
    flip_vertical_p=0.5,
    affine_p=1.,
    affine_scale_choices=np.linspace(0.5, 1.5, 11),
    affine_shear_choices=np.linspace(-0.1, 0.1, 11),
    affine_translation_choices=np.arange(-64, 64, 1),
    affine_rotation_choices=np.arange(-30, 30, 1),
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
    W=nn.init.GlorotNormal(gain=1 / 3.0),
)

l = nn.layers.InputLayer(name='in', shape=(None, 3, image_size, image_size))

l = conv2dbn(l, name='1c1', num_filters=32, filter_size=3, stride=2)
l = conv2dbn(l, name='1c2', num_filters=32, filter_size=3)
l = conv2dbn(l, name='1c3', num_filters=64, filter_size=3, pad=1)
l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='1p', pool_size=3, stride=2)
l = nn.layers.DropoutLayer(l, name='1cdrop', p=0.1)

l = conv2dbn(l, name='2c1', num_filters=80, filter_size=1)
l = conv2dbn(l, name='2c2', num_filters=192, filter_size=3)
l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='2p', pool_size=3, stride=2)
l = nn.layers.DropoutLayer(l, name='2cdrop', p=0.1)

l = inceptionA(
    l, name='3', nfilt=(
        (64,),
        (48, 64),
        (64, 96, 96),
        (32,)
    )
)
l = nn.layers.DropoutLayer(l, name='3cdrop', p=0.1)
l = inceptionA(
    l, name='4', nfilt=(
        (64,),
        (48, 64),
        (64, 96, 96),
        (64,)
    )
)
l = nn.layers.DropoutLayer(l, name='4cdrop', p=0.1)

l = inceptionB(
    l, name='5', nfilt=(
        (384,),
        (64, 96, 96)
    )
)
l = nn.layers.DropoutLayer(l, name='5cdrop', p=0.1)

l = inceptionC(
    l, name='6', nfilt=(
        (192,),
        (128, 128, 192),
        (128, 128, 128, 128, 192),
        (192,)
    )
)
l = nn.layers.DropoutLayer(l, name='6cdrop', p=0.1)

l = inceptionC(
    l, name='7', nfilt=(
        (192,),
        (192, 192, 192),
        (192, 192, 192, 192, 192),
        (192,)
    )
)
l = nn.layers.DropoutLayer(l, name='7cdrop', p=0.1)

l = inceptionD(
    l, name='8', nfilt=(
        (192, 320),
        (192, 192, 192, 192)
    )
)
l = nn.layers.DropoutLayer(l, name='8cdrop', p=0.1)

l = inceptionE(
    l, name='9', nfilt=(
        (320,),
        (384, 384, 384),
        (448, 384, 384, 384),
        (192,)
    ),
    pool_type='avg'
)
l = nn.layers.DropoutLayer(l, name='9cdrop', p=0.1)

l = inceptionE(
    l, name='11', nfilt=(
        (320,),
        (384, 384, 384),
        (448, 384, 384, 384),
        (192,)
    ),
    pool_type='max'
)

l = nn.layers.GlobalPoolLayer(l, name='gp')
l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.8)

l = nn.layers.DenseLayer(l, name='out', num_units=n_classes, nonlinearity=nn.nonlinearities.softmax)


net = NeuralNet(
    layers=l,

    regression=False,
    use_label_encoder=False,

    objective_l2=1e-6,

    update=nn.updates.nesterov_momentum,
    update_learning_rate=theano.shared(float32(1e-1)),

    train_split=TrainSplit(0.15, random_state=42, stratify=False),
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping,
        StepDecay('update_learning_rate', start=1e-1, stop=1e-3)
    ],

    verbose=10,

    max_epochs=2500,
)
