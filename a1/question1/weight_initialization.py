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
    """
    Initialisation scheme that sets all weights to 0
    """
    @staticmethod
    def init(dim_l, dim_l_prev):
        """
        Returns a dim_l x dim_l_prev dimensional W matrix
        """
        return np.zeros((dim_l, dim_l_prev))


class Normal(object):
    """
    Initialisation scheme that sets to the standard normal distribution (mean=0, variance=1)
    """
    @staticmethod
    def init(dim_l, dim_l_prev):
        """
        Returns a dim_l x dim_l_prev dimensional W matrix
        """
        return np.random.randn(dim_l, dim_l_prev)


class ScaledNormal(object):
    """
        Initialisation scheme that sets to the standard normal distribution (mean=0, variance=1) with scaling
        depending on the dimension of the input layer to reduce weights closer to 0
        """
    @staticmethod
    def init(dim_l, dim_l_prev):
        """
        Returns a dim_l x dim_l_prev dimensional W matrix
        """
        return np.random.randn(dim_l, dim_l_prev) * np.sqrt(1. / dim_l_prev)


class Glorot(object):
    """
    Initialisation scheme that sets weights to a uniform distribution [-d^(l), d^(l)]
    where d^(l) = sqrt(6/(dim_l-1+dim_l))
    """
    @staticmethod
    def init(dim_l, dim_l_prev):
        """
        Returns a dim_l x dim_l_prev dimensional W matrix
        """
        d_l = np.sqrt(6.0/(dim_l+dim_l_prev))
        return np.random.uniform(-d_l, d_l, (dim_l, dim_l_prev))
