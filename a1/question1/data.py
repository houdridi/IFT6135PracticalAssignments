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
import gzip
import pickle
import pandas as pd
import os
import urllib
from threading import Lock

DATASETS_DIR = 'datasets'
MNIST_FILE_NAME = 'mnist.pkl.gz'
MNIST_URL = 'http://deeplearning.net/data/mnist'
DIGITS = 10


def download_mnist():
    """
    Downloads the mnist dataset (if it does not already exist)
    :return: Raw mnist dataset
    """
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    save_path = os.path.join(DATASETS_DIR, MNIST_FILE_NAME)

    if not os.path.exists(save_path):
        print("Downloading MNIST dataset from %s" % MNIST_URL)
        urllib.urlretrieve("{}/{}".format(MNIST_URL, MNIST_FILE_NAME), save_path)

    with gzip.open(save_path, 'r') as f:
        return pickle.load(f, encoding='iso-8859-1')


def preprocess(data_set):
    """
    Pre-processes (x,y) values. Encodes y values as one-hot vectors.
    Arranges x and y matrices such that samples are arranged by column
    :param data_set: Dataset in (x,y) tuple format
    :return: Pre-processes dataset
    """
    x, y = data_set
    return x.T, np.eye(DIGITS)[y].T


def load_mnist():
    """
    Loads and returns the mnist dataset in train_set, validation-set, test_set formats.
    Each set is a (x,y) tuple
    """
    train_set, validation_set, test_set = download_mnist()
    return preprocess(train_set), preprocess(validation_set), preprocess(test_set)


class ResultsCache(object):
    """
    Utility class for storing results of hyper-parameter search
    """
    FILE_PATH = './results/params_search.csv'

    def __init__(self, df):
        self.df = df
        self.save_lock = Lock()

    @staticmethod
    def load():
        """
        Loads the results cache. If previous results stored in the .csv file
        exist on the file-system, loads them into memory.
        """
        df = pd.read_csv(ResultsCache.FILE_PATH) if os.path.isfile(
            ResultsCache.FILE_PATH) else pd.DataFrame(
            columns=['label', 'activation', 'weight_init', 'layers', 'alpha', 'batch', 'acc'])
        return ResultsCache(df)

    def display(self):
        """
        Displays all results stored in the result cache
        """
        print('\nParameter Search Results Summary:')
        print(self.df.drop('label', 1))

    def insert(self, nn, alpha, batch, acc):
        """
        Inserts a new result into the result cache
        :param nn: Neural network model
        :param alpha: Learning rate used to train the model
        :param batch: Batch size used to train the model
        :param acc: Validation accuracy obtained after training
        """
        label = nn.get_training_info_str(alpha, batch).replace(u'\u03B1', 'alpha').replace(' ', '')

        self.save_lock.acquire()
        try:
            if not self.df.loc[self.df['label'] == label].empty:
                # Update existing entry
                self.df.loc[self.df['label'] == label, 'acc'] = acc
            else:
                # Insert new entry
                new_entry = {'label': label, 'activation': nn.activation.__name__,
                             'weight_init': nn.weight_init.__name__,
                             'layers': '-'.join([str(l) for l in nn.layer_dims]),
                             'alpha': alpha, 'batch': batch, 'acc': acc}
                self.df = self.df.append(new_entry, ignore_index=True)

            self.df = self.df.sort_values(by=['acc'], ascending=False)
            self.df.to_csv(ResultsCache.FILE_PATH, index=False)

        finally:
            self.save_lock.release()

