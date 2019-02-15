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
