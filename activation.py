import numpy as np


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """
    Derivate of the sigmoid function.

    :param y:
    :return:
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    """
    Hyperbolic tangent

    :param x:
    :return:
    """
    return np.tanh(x)


def tanh_der(x):
    """
    Derivate of the hyperbolic tangent function.

    :param x:
    :return:
    """
    return 1.0 - np.power(tanh(x), 2)


fun_dict = {'sigmoid': {'activation': sigmoid,
                        'derivative': sigmoid_der},
            'tanh': {'activation': tanh,
                     'derivative': tanh_der},
            'linear': {'activation': lambda x: x,
                       'derivative': lambda x: 1.0}}

