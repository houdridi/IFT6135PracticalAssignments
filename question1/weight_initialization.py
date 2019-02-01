import numpy as np


class Zeros(object):
    @staticmethod
    def init(dim_l, dim_l_prev):
        return np.zeros((dim_l, dim_l_prev))


class Normal(object):

    @staticmethod
    def init(dim_l, dim_l_prev):
        return np.random.randn(dim_l, dim_l_prev) * np.sqrt(1. / dim_l_prev)


class Glorot(object):
    @staticmethod
    def init(dim_l, dim_l_prev):
        pass
