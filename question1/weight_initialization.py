"""
IFT 6135 - W2019 - Practical Assignment 1 - Question 1
Assignment Instructions: https://www.overleaf.com/read/msxwmbbvfxrd
Github Repository: https://github.com/stefanwapnick/IFT6135PracticalAssignments
Developed in Python 3

Mohamed Amine (UdeM ID: 20150893)
Oussema Keskes (UdeM ID: 20145195)
Stephan Tran (UdeM ID: 20145195)
Stefan Wapnick (UdeM ID: 20143021)
"""
import numpy as np


class Zeros(object):
    @staticmethod
    def init(dim_l, dim_l_prev):
        return np.zeros((dim_l, dim_l_prev))


class Normal(object):

    @staticmethod
    def init(dim_l, dim_l_prev):
        return np.random.randn(dim_l, dim_l_prev)


class ScaledNormal(object):

    @staticmethod
    def init(dim_l, dim_l_prev):
        return np.random.randn(dim_l, dim_l_prev) * np.sqrt(1. / dim_l_prev)


class Glorot(object):
    @staticmethod
    def init(dim_l, dim_l_prev):
        d_l = np.sqrt(6.0/(dim_l+dim_l_prev))
        return np.random.uniform(-d_l, d_l, (dim_l, dim_l_prev))
