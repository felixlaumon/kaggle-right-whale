import lasagne as nn
from lasagne.layers import dnn
from utils.layers import batch_norm2 as batch_norm


def conv2dbn(l, name, **kwargs):
    """ Batch normalized DNN Conv2D Layer """
    l = nn.layers.dnn.Conv2DDNNLayer(
        l, name=name,
        **kwargs
    )
    l = batch_norm(l, name='%sbn' % name)
    return l


def conv2dbn2(l, name='', **kwargs):
    """ Batch normalized DNN Conv2D Layer """
    l = nn.layers.dnn.Conv2DDNNLayer(
        l, name=name,
        **kwargs
    )
    l = nn.layers.batch_norm(l, name='%sbn' % name)
    return l


# def residual_block(layer, name, num_layers,
#                    num_filters, filter_size=3, stride=1, pad='same',
#                    nonlinearity=nn.nonlinearities.rectify):
#     conv = layer
#     if (num_filters != layer.output_shape[1]) or (stride != 1):
#         layer = conv2dbn(
#             layer, name='%s_shortcut' % name, num_filters=num_filters,
#             filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None
#         )

#     for i in range(num_layers):
#         conv = conv2dbn(
#             conv, name='%s_%s' % (name, i), num_filters=num_filters,
#             filter_size=filter_size, pad=pad,
#             # Remove nonlinearity for the last conv layer
#             nonlinearity=nonlinearity if (i == num_layers - 1) else None,
#             # Only apply stride for the first conv layer
#             stride=stride if i == 0 else 1
#         )

#     l = nn.layers.merge.ElemwiseSumLayer([conv, layer], name='%s_merge' % name)
#     l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity, name='%s_merge_nl' % name)
#     return l


def residual_block(layer, name, num_layers, num_filters,
                   bottleneck=False, bottleneck_factor=4,
                   filter_size=(3, 3), stride=1, pad='same',
                   W=nn.init.GlorotUniform(),
                   nonlinearity=nn.nonlinearities.rectify):
    conv = layer

    # When changing filter size or feature map size
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        # Projection shortcut, aka option B
        layer = conv2dbn(
            layer, name='%s_shortcut' % name, num_filters=num_filters,
            filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None
        )

    if bottleneck and num_layers < 3:
        raise ValueError('At least 3 layers is required for bottleneck configuration')

    for i in range(num_layers):
        if bottleneck:
            # Force then first and last layer to use 1x1 convolution
            if i == 0 or (i == (num_layers - 1)):
                actual_filter_size = (1, 1)
            else:
                actual_filter_size = filter_size

            # Only increase the filter size to the target size for
            # the last layer
            if i == (num_layers - 1):
                actual_num_filters = num_filters
            else:
                actual_num_filters = num_filters / bottleneck_factor
        else:
            actual_num_filters = num_filters
            actual_filter_size = filter_size

        conv = conv2dbn(
            conv, name='%s_%s' % (name, i), num_filters=actual_num_filters,
            filter_size=actual_filter_size, pad=pad, W=W,
            # Remove nonlinearity for the last conv layer
            nonlinearity=nonlinearity if (i < num_layers - 1) else None,
            # Only apply stride for the first conv layer
            stride=stride if i == 0 else 1
        )

    l = nn.layers.merge.ElemwiseSumLayer([conv, layer], name='%s_elemsum' % name)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity, name='%s_elemsum_nl' % name)
    return l


# TODO WTF is localbn? is this different from residual_block3?
def residual_block3_localbn(layer, name, num_layers, num_filters,
                            bottleneck=False, bottleneck_factor=4,
                            filter_size=(3, 3), stride=1, pad='same',
                            W=nn.init.GlorotUniform(),
                            nonlinearity=nn.nonlinearities.rectify):
    conv = layer

    # Insert shortcut when changing filter size or feature map size
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        # Projection shortcut, aka option B
        layer = nn.layers.dnn.Conv2DDNNLayer(
            layer, name='%s_shortcut' % name, num_filters=num_filters,
            filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None
        )

    if bottleneck and num_layers < 3:
        raise ValueError('At least 3 layers is required for bottleneck configuration')

    for i in range(num_layers):
        if bottleneck:
            # Force then first and last layer to use 1x1 convolution
            if i == 0 or (i == (num_layers - 1)):
                actual_filter_size = (1, 1)
            else:
                actual_filter_size = filter_size

            # Only increase the filter size to the target size for
            # the last layer
            if i == (num_layers - 1):
                actual_num_filters = num_filters
            else:
                actual_num_filters = num_filters / bottleneck_factor
        else:
            actual_num_filters = num_filters
            actual_filter_size = filter_size

        # TODO the last layer should probably not be bn-ed..
        conv = conv2dbn(
            conv, name='%s_%s' % (name, i), num_filters=actual_num_filters,
            filter_size=actual_filter_size, pad=pad, W=W,
            # Remove nonlinearity for the last conv layer
            nonlinearity=nonlinearity if (i < num_layers - 1) else None,
            # Only apply stride for the first conv layer
            stride=stride if i == 0 else 1
        )

    l = nn.layers.merge.ElemwiseSumLayer([conv, layer], name='%s_elemsum' % name)
    l = batch_norm(l)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity, name='%s_elemsum_nl' % name)
    return l


def residual_block3(layer, name, num_layers, num_filters,
                    bottleneck=False, bottleneck_factor=4,
                    filter_size=(3, 3), stride=1, pad='same',
                    W=nn.init.GlorotUniform(),
                    nonlinearity=nn.nonlinearities.rectify):
    conv = layer

    # Insert shortcut when changing filter size or feature map size
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        # Projection shortcut, aka option B
        layer = nn.layers.dnn.Conv2DDNNLayer(
            layer, name='%s_shortcut' % name, num_filters=num_filters,
            filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None
        )

    if bottleneck and num_layers < 3:
        raise ValueError('At least 3 layers is required for bottleneck configuration')

    for i in range(num_layers):
        if bottleneck:
            # Force then first and last layer to use 1x1 convolution
            if i == 0 or (i == (num_layers - 1)):
                actual_filter_size = (1, 1)
            else:
                actual_filter_size = filter_size

            # Only increase the filter size to the target size for
            # the last layer
            if i == (num_layers - 1):
                actual_num_filters = num_filters
            else:
                actual_num_filters = num_filters / bottleneck_factor
        else:
            actual_num_filters = num_filters
            actual_filter_size = filter_size

        # TODO the last layer should probably not be bn-ed..
        conv = conv2dbn2(
            conv, name='%s_%s' % (name, i), num_filters=actual_num_filters,
            filter_size=actual_filter_size, pad=pad, W=W,
            # Remove nonlinearity for the last conv layer
            nonlinearity=nonlinearity if (i < num_layers - 1) else None,
            # Only apply stride for the first conv layer
            stride=stride if i == 0 else 1
        )

    l = nn.layers.merge.ElemwiseSumLayer([conv, layer], name='%s_elemsum' % name)
    l = nn.layers.batch_norm(l)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity, name='%s_elemsum_nl' % name)
    return l
