import numpy as np


class Tanh(object):

    @staticmethod
    def f(z):
        exp_plus = np.exp(z)
        exp_neg = 1. / exp_plus
        return (exp_plus - exp_neg) / (exp_plus + exp_neg)

    @staticmethod
    def df(z):
        tanh = Tanh.f(z)
        return 1. - tanh * tanh


class Sigmoid(object):

    @staticmethod
    def f(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def df(z):
        sig = Sigmoid.f(z)
        return sig * (1 - sig)


class Relu(object):

    @staticmethod
    def f(z):
        return np.maximum(0, z)

    @staticmethod
    def df(z):
        return (z > 0).astype(int)
