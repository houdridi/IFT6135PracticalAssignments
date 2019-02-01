import urllib
import cPickle as pickle
import gzip
import os
import numpy as np
import argparse

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='mnist',
    #                     choices=['mnist'],
    #                     help='dataset name')
    # parser.add_argument('--savedir', type=str, default='datasets',
    #                     help='directory to save the dataset')

    # args = parser.parse_args()
    # print(args)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    path = 'http://deeplearning.net/data/mnist'
    mnist_filename_all = 'mnist.pkl'
    local_filename = os.path.join('datasets', mnist_filename_all)
    urllib.urlretrieve(
        "{}/{}.gz".format(path, mnist_filename_all), local_filename + '.gz')
    tr, va, te = pickle.load(gzip.open(local_filename + '.gz', 'r'))
