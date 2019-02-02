"""
IFT 6135 - W2019 - Practical Assignment 1 - Question 1
Assignment Instructions: https://www.overleaf.com/read/msxwmbbvfxrd
Github Repository: https://github.com/stefanwapnick/IFT6135PracticalAssignments
"""
import urllib
import cPickle as pickle
import gzip
import os
import numpy as np
import argparse

if __name__ == '__main__':

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    path = 'http://deeplearning.net/data/mnist'
    mnist_filename_all = 'mnist.pkl'
    local_filename = os.path.join('datasets', mnist_filename_all)
    urllib.urlretrieve(
        "{}/{}.gz".format(path, mnist_filename_all), local_filename + '.gz')
