import theano
import theano.tensor as T
import lasagne as nn


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng = RandomStreams()


class TiedDropoutLayer(nn.layers.Layer):
    """
    Dropout layer that broadcasts the mask across all axes beyond the first two.
    """
    def __init__(self, input_layer, p=0.5, rescale=True, **kwargs):
        super(TiedDropoutLayer, self).__init__(input_layer, **kwargs)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            mask = _srng.binomial(input.shape[:2], p=retain_prob,
                                  dtype=theano.config.floatX)
            axes = [0, 1] + (['x'] * (input.ndim - 2))
            mask = mask.dimshuffle(*axes)
            return input * mask


class BatchNormLayer(nn.layers.Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.5, nonlinearity=None, mode='low_mem',
    beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), var=lasagne.init.Constant(1), **kwargs)
    Batch Normalization
    This layer implements batch normalization of its inputs, following [1]_:
    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta
    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed.
    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`.
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numeric problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If ``None``
        is provided, the layer will be linear (this is the default).
    mode : {'low_mem', 'high_mem'}
        Specify which batch normalization implementation to use: ``'low_mem'``
        avoids storing intermediate representations and thus requires less
        memory, while ``'high_mem'`` can reuse representations for the backward
        pass and is thus 5-10% faster.
    beta : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    var : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\sigma^2`. Must
        match the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.
    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.
    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.
    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.
    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.
    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer
    References
    ----------
    .. [1]: Ioffe, Sergey and Szegedy, Christian (2015):
            Batch Normalization: Accelerating Deep Network Training by Reducing
            Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 nonlinearity=None, mode='low_mem',
                 beta=nn.init.Constant(0), gamma=nn.init.Constant(1),
                 mean=nn.init.Constant(0), var=nn.init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = nn.nonlinearities.identity
        self.nonlinearity = nonlinearity
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        self.beta = self.add_param(beta, shape, 'beta',
                                   trainable=True, regularizable=False)
        self.gamma = self.add_param(gamma, shape, 'gamma',
                                    trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.var = self.add_param(var, shape, 'var',
                                  trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_var = input.var(self.axes)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = kwargs.get('batch_norm_use_averages',
                                  deterministic)
        if use_averages:
            mean = self.mean
            var = self.var
        else:
            mean = input_mean
            var = input_var

        # Decide whether to update the stored averages
        update_averages = kwargs.get('batch_norm_update_averages',
                                     not deterministic)
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_var.default_update = ((1 - self.alpha) * running_var +
                                          self.alpha * input_var)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            var += 0 * running_var

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(self.beta.ndim))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = self.beta.dimshuffle(pattern)
        gamma = self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = T.sqrt(var + self.epsilon)
        std = std.dimshuffle(pattern)

        # normalize
        # normalized = (input - mean) * (gamma / std) + beta
        normalized = T.nnet.batch_normalization(input, gamma=gamma, beta=beta,
                                                mean=mean, std=std,
                                                mode=self.mode)
        return self.nonlinearity(normalized)


def batch_norm(layer, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` on top.
    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor. Especially note the `mode`
        argument, which controls a memory usage to performance tradeoff.
    Returns
    -------
    :class:`BatchNormLayer` instance
        A batch normalization layer stacked on the given modified `layer`.
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nn.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity, **kwargs)


def batch_norm2(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nn.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        from lasagne.layers.special import NonlinearityLayer
        layer = NonlinearityLayer(layer, nonlinearity, name='%s_nl' % layer.name)
    return layer
