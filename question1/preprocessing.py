import numpy as np
import gzip
import cPickle as pickle


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
