import lasagne as nn
import theano.tensor as T
from utils.layer_macros import conv2dbn


def avg_pool(input_layer, **kwargs):
    # hack to work around https://github.com/Theano/Theano/issues/3776
    norm = nn.layers.ExpressionLayer(input_layer, lambda X: T.ones_like(X))
    norm = nn.layers.Pool2DLayer(norm, mode='average_inc_pad', **kwargs)
    l = nn.layers.Pool2DLayer(input_layer, mode='average_inc_pad', **kwargs)
    l = nn.layers.ElemwiseMergeLayer([l, norm], T.true_div)
    return l


def inceptionA(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptA_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptA_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptA_2_5x5' % name,
        num_filters=nfilt[1][1], filter_size=5, pad=2
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptA_3_1x1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptA_3_3x3_1' % name,
        num_filters=nfilt[2][1], filter_size=3, pad=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptA_3_3x3_2' % name,
        num_filters=nfilt[2][2], filter_size=3, pad=1
    )

    l4 = avg_pool(
        input_layer, name='%s_inceptA_4p' % name,
        pool_size=3, stride=1, pad=1,
    )
    l4 = conv2dbn(
        l4, name='%s_inceptA_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3, l4], name='%s_inceptA_concat' % name
    )


def inceptionB(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptB_1_3x3' % name,
        num_filters=nfilt[0][0], filter_size=3, stride=2
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptB_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptB_2_3x3_1' % name,
        num_filters=nfilt[1][1], filter_size=3, pad=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptB_2_3x3_2' % name,
        num_filters=nfilt[1][2], filter_size=3, stride=2
    )

    l3 = avg_pool(
        input_layer, name='%s_inceptB_3p' % name,
        pool_size=3, stride=2,
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3], name='%s_inceptB_concat' % name
    )


def inceptionC(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptC_1_3x3' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptC_2_3x3' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptC_2_1x7' % name,
        num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptC_2_7x1' % name,
        num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0)
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptC_3_1x1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_7x1_1' % name,
        num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_1x7_1' % name,
        num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_7x1_2' % name,
        num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_1x7_2' % name,
        num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3)
    )

    l4 = avg_pool(
        input_layer, name='%s_inceptC_4p' % name,
        pool_size=3, stride=1, pad=1
    )
    l4 = conv2dbn(
        l4, name='%s_inceptC_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3, l4], name='%s_inceptC_concat' % name
    )


def inceptionD(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptD_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )
    l1 = conv2dbn(
        l1, name='%s_inceptD_1_3x3' % name,
        num_filters=nfilt[0][1], filter_size=3, stride=2
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptD_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_1x7' % name,
        num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_7x1' % name,
        num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_3x3' % name,
        num_filters=nfilt[1][3], filter_size=3, stride=2
    )

    l3 = nn.layers.dnn.Pool2DDNNLayer(
        input_layer, name='%s_inceptD_3p' % name,
        pool_size=3, stride=2, mode='max'
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3], name='%s_inceptD_concat' % name
    )


def inceptionE(input_layer, name, nfilt, pool_type):
    l1 = conv2dbn(
        input_layer, name='%s_inceptE_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptE_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2a = conv2dbn(
        l2, name='%s_inceptE_2a_1x3' % name,
        num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1)
    )
    l2b = conv2dbn(
        l2, name='%s_inceptE_2b_3x1' % name,
        num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0)
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptE_3_1x1_1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptE_3_1x1_2' % name,
        num_filters=nfilt[2][1], filter_size=3, pad=1
    )
    l3a = conv2dbn(
        l3, name='%s_inceptE_3a_1x3' % name,
        num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1)
    )
    l3b = conv2dbn(
        l3, name='%s_inceptE_3b_3x1' % name,
        num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0)
    )

    if pool_type == 'avg':
        l4 = avg_pool(
            input_layer, name='%s_inceptE_4p' % name,
            pool_size=3, stride=1, pad=1
        )
    elif pool_type == 'max':
        l4 = nn.layers.dnn.Pool2DDNNLayer(
            input_layer, name='%s_inceptE_4p' % name,
            pool_size=3, stride=1, pad=1, mode='max'
        )
    else:
        raise ValueError('unrecognized pool_type')
    l4 = conv2dbn(
        l4, name='%s_inceptE_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2a, l2b, l3a, l3b, l4], name='%s_inceptE_concat' % name
    )
