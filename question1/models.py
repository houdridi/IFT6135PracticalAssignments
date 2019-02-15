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

from activations import Sigmoid
from weight_initialization import Normal, Glorot
from time import time
from sklearn.metrics import accuracy_score


class NN(object):

    def __init__(self, layers_dims, activation=Sigmoid, weight_init=Glorot):
        self.b = {}
        self.w = {}
        self.activation = activation
        self.layer_dims = layers_dims
        self.layers = len(self.layer_dims)
        self.weight_init = weight_init

    def loss(self, y, ypred):
        m = y.shape[1]
        return -(1. / m) * np.sum(np.multiply(y, np.log(ypred)))

    def initialize_weights(self):
        for l in range(1, len(self.layer_dims)):
            dim_l_prev, dim_l = self.layer_dims[l - 1], self.layer_dims[l]
            self.w[l] = self.weight_init.init(dim_l, dim_l_prev)
            self.b[l] = np.zeros((dim_l, 1))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def forward(self, X):
        z, a = {}, {}

        a[0] = X
        for l in range(1, self.layers):
            z[l] = np.matmul(self.w[l], a[l - 1]) + self.b[l]
            a[l] = self.activation.f(z[l]) if l < self.layers - 1 else self.softmax(z[l])

        return z, a

    def backward(self, z, a, y):
        dz, da, dw, db = {}, {}, {}, {}
        m_batch = y.shape[1]

        for l in reversed(range(1, self.layers)):
            dz[l] = np.multiply(da[l], self.activation.df(z[l])) if l < self.layers - 1 else a[self.layers - 1] - y
            dw[l] = (1. / m_batch) * np.matmul(dz[l], a[l - 1].T)
            db[l] = (1. / m_batch) * np.sum(dz[l], axis=1, keepdims=True)
            da[l - 1] = np.matmul(self.w[l].T, dz[l]) if l > 1 else 0
        return dw, db

    def update(self, dw, db, alpha):
        for l in range(1, self.layers):
            self.w[l] = self.w[l] - alpha * dw[l]
            self.b[l] = self.b[l] - alpha * db[l]

    def test(self, x, y):
        z, a = self.forward(x)
        y_pred = a[self.layers - 1]
        cost = self.loss(y, y_pred)
        accuracy = accuracy_score(np.argmax(y, axis=0), np.argmax(y_pred, axis=0))
        return cost, accuracy

    def get_training_info_str(self, alpha, batch_size):
        return u'g=%s, w_init=%s, layers=%s, \u03B1=%.2f, batch=%d' \
               % (self.activation.__name__, self.weight_init.__name__,
                  '-'.join([str(l) for l in self.layer_dims]), alpha, batch_size)

    def estimate_finite_diff_gradient(self, x_sample, y_sample, eps, weight_index):
        # Get coordinates of weight to estimate finite difference for: weight_index = (layer_idx, neuron_idx, input_idx)
        # Recall weight matrix is NxM where N = # of neurons in l'th layer, M = number of inputs from previous layer
        # Weight can be indexed by layer_idx, neuron_idx, input_idx
        (layer_idx, neuron_idx, input_idx) = weight_index
        weight_i = self.w[layer_idx][neuron_idx][input_idx]

        self.w[layer_idx][neuron_idx][input_idx] = weight_i - eps
        z, a = self.forward(x_sample)
        train_cost_neg_eps = self.loss(y_sample, a[self.layers - 1])

        self.w[layer_idx][neuron_idx][input_idx] = weight_i + eps
        z, a = self.forward(x_sample)
        train_cost_pos_eps = self.loss(y_sample, a[self.layers - 1])

        self.w[layer_idx][neuron_idx][input_idx] = weight_i
        z, a = self.forward(x_sample)
        dw, db = self.backward(z, a, y_sample)
        return abs((train_cost_pos_eps - train_cost_neg_eps) / (2 * eps) - dw[layer_idx][neuron_idx][input_idx])

    def train(self, train_set, valid_set, alpha=0.1, epochs=10, batch_size=128, verbose=True):

        self.initialize_weights()
        x_train, y_train = train_set
        x_valid, y_valid = valid_set
        m = x_train.shape[1]
        n_batches = int(np.ceil(float(m) / batch_size))
        train_lost = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        valid_lost = np.zeros(epochs)
        valid_acc = np.zeros(epochs)

        start = time()
        label = self.get_training_info_str(alpha, batch_size)
        if verbose:
            print("TRAINING: %s" % label)

        for i in range(epochs):
            rand_order = np.random.permutation(m)
            x_shuffled = x_train[:, rand_order]
            y_shuffled = y_train[:, rand_order]

            for k in range(n_batches):
                begin = k * batch_size
                end = min(begin + batch_size, m - 1)
                x_batch = x_shuffled[:, begin:end]
                y_batch = y_shuffled[:, begin:end]

                z, a = self.forward(x_batch)
                dw, db = self.backward(z, a, y_batch)
                self.update(dw, db, alpha)

            train_lost[i], train_acc[i] = self.test(x_train, y_train)
            valid_lost[i], valid_acc[i] = self.test(x_valid, y_valid)

            if verbose:
                print("Epoch %d: TrainLoss=%f, TrainAcc=%f, ValidLoss=%f, ValidAcc=%f"
                      % (i + 1, train_lost[i], train_acc[i], valid_lost[i], valid_acc[i]))

        print("DONE after %ds: %s - ValidLoss=%f, ValidAcc=%f" % (time() - start, label,
                                                                                 valid_lost[-1], valid_acc[-1]))
        return train_lost, train_acc, valid_lost, valid_acc
