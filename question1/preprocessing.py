import numpy as np
import gzip
import cPickle as pickle
import pandas as pd
import os
from threading import Lock


def load_mnist():
    with gzip.open('datasets/mnist.pkl.gz', 'r') as f:
        tr, va, te = pickle.load(f)

    X_train = tr[0].T
    X_test = va[0].T
    y_train = tr[1]
    y_test = va[1]

    digits = 10
    examples = y_train.shape[0]
    test_examples = y_test.shape[0]

    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])

    y_train = np.eye(digits)[y_train.astype('int32')]
    y_train = y_train.T.reshape(digits, examples)

    y_test = np.eye(digits)[y_test.astype('int32')]
    y_test = y_test.T.reshape(digits, test_examples)
    return X_train, y_train, X_test, y_test


class ParamsSearchResults(object):
    FILE_PATH = './results/params_search.csv'

    def __init__(self, df):
        self.df = df
        self.save_lock = Lock()

    @staticmethod
    def load():
        df = pd.read_csv(ParamsSearchResults.FILE_PATH) if os.path.isfile(
            ParamsSearchResults.FILE_PATH) else pd.DataFrame(
            columns=['label', 'activation', 'weight_init', 'layers', 'alpha', 'batch', 'acc'])
        return ParamsSearchResults(df)

    def display(self):
        print(self.df.drop('label', 1))

    def insert(self, nn, alpha, batch, acc):
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
            self.df.to_csv(ParamsSearchResults.FILE_PATH, index=False)

        finally:
            self.save_lock.release()

