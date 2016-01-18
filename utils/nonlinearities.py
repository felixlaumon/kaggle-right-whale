import theano
import theano.tensor as T


# Taken from https://github.com/krallistic/Lasagne/blob/temperature_Softmax/lasagne/nonlinearities.py
# temperature softmax
class TemperatureSoftmax(object):
    """Temperature Softmax :math:`q_{i} = \\frac{e^{(z_i / T)}}{\\sum_j e^{(z_i / T)}}
    Compared to the standard  softmax :func:`softmax`, allows to change the difference in selection probability for
    outputs that differ in their value estimates.
    ----------
    temperature : float
        temperature for softmax calculation, usually between 0 and +\\infty.
        a temperature of 1 will lead to the Standart Softmax
        a temperature >1 will lead to a softer probability distribution over classes,
        a temperature <1 will lead to a harder probability distribution over classes
    Methods
    -------
    __call__(x)
        Apply the softmax function to the activation `x`.
    Examples
    --------
    In contrast to most other activation functions in this module, this is
    a class that needs to be instantiated to obtain a callable:
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((None, 100))
    >>> from lasagne.nonlinearities import TemperatureSoftmax()
    >>> custom_softmax = TemperatureSoftmax(0.1)
    >>> l1 = DenseLayer(l_in, num_units=200, nonlinearity=custom_softmax)
    Alternatively, you can use the provided instance for temperature=0.1:
    >>> from lasagne.nonlinearities import low_temperature_softmax
    >>> l2 = DenseLayer(l_in, num_units=200, nonlinearity=low_temperature_softmax)
    See Also
    --------
    low_temperature_softmax: Instance with default temperature of 0.01, as in [1]_.
    high_temperature_softmax: Instance with high temperature of 3, as in [2]_.
    References
    ----------
    .. [1] TODO
    .. [2] Geoffrey Hinton, Oriol Vinyals, Jeff Dean:
       Distilling the Knowledge in a Neural Network
       http://arxiv.org/abs/1503.02531
    """

    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def __call__(self, x):
        if self.temperature != 1:
            e_x = T.exp(x / self.temperature)
            return (e_x) / (e_x.sum(axis=-1).dimshuffle(0, 'x'))
        else:
            return theano.tensor.nnet.softmax(x)


low_temperature_softmax = TemperatureSoftmax(temperature=0.1)
high_temperature_softmax = TemperatureSoftmax(temperature=3)
