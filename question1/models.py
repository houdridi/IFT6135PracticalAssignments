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
    """
    Feed-forward neural network (multi-layer perceptron) for multi-class classification.
    Loss computing using cross-entropy. Optimized by stochastic gradient descent with mini-batches.
    """
    def __init__(self, layers_dims, activation=Sigmoid, weight_init=Glorot):
        """
        Creates a new neural network instance
        :param layers_dims: Dimensions of neural network. Includes input, hidden and output layer sizes
        :param activation: Activation function: [Sigmoid, Tanh, Relu]
        :param weight_init: Weight initialization scheme: [Zeros, Normal, Glorot]
        """
        self.b = {}
        self.w = {}
        self.activation = activation
        self.layer_dims = layers_dims
        self.layers = len(self.layer_dims)
        self.weight_init = weight_init
        self.training_info_label = ''

    def loss(self, y, ypred):
        """
        Computes the cross entropy loss between predicted and labeled classification results
        :param y: True y values
        :param ypred: Predicted y values
        :return: Multi-class Cross-entropy loss
        """
        m = y.shape[1]
        return -(1. / m) * np.sum(np.multiply(y, np.log(ypred)))

    def initialize_weights(self):
        """
        Initialize weights among all layers according to the weight initializer scheme provided when
        the neural network was created
        """
        for l in range(1, len(self.layer_dims)):
            dim_l_prev, dim_l = self.layer_dims[l - 1], self.layer_dims[l]
            self.w[l] = self.weight_init.init(dim_l, dim_l_prev)
            self.b[l] = np.zeros((dim_l, 1))

    def softmax(self, z):
        """
        Computes the softmax classification probability for each class
        :param z: Pre-activation of final layer of neural network, each element corresponding to a possible class.
        :return: Classification probability of each class
        """
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def forward(self, X):
        """
        Executes a forward pass through the neural network
        :param X: Training data N x M matrix (N = feature vector dimension, M = # samples)
        :return: History of pre-activation (z) and post-activation (a) values
        """
        z, a = {}, {}

        a[0] = X
        for l in range(1, self.layers):
            z[l] = np.matmul(self.w[l], a[l - 1]) + self.b[l]
            # If last layer, output is computed using softmax, else use normal activation function configured with nn
            a[l] = self.activation.f(z[l]) if l < self.layers - 1 else self.softmax(z[l])

        return z, a

    def backward(self, z, a, y):
        """
        Executes backward propagation. Recursively computes dL/dz^(l), dl/da^(l), dL/dw^(l), dl/db^(l) balues
        :param z: History of pre-activation values from the forward pass
        :param a: History of post-activation values from the forward pass
        :param y: Labeled (true) y values for training data
        :return: Gradient values for weights (dw) and biases (db) from which weights can be adjusted by gradient descent
        """
        dz, da, dw, db = {}, {}, {}, {}
        m_batch = y.shape[1]

        for l in reversed(range(1, self.layers)):
            # If last layer compute gradient using softmax activation function. Else compute gradient using derivative
            # of normal activation function
            dz[l] = np.multiply(da[l], self.activation.df(z[l])) if l < self.layers - 1 else a[self.layers - 1] - y
            dw[l] = (1. / m_batch) * np.matmul(dz[l], a[l - 1].T)
            db[l] = (1. / m_batch) * np.sum(dz[l], axis=1, keepdims=True)
            da[l - 1] = np.matmul(self.w[l].T, dz[l]) if l > 1 else 0
        return dw, db

    def update(self, dw, db, alpha):
        """
        Updates weight values using gradients from backward propagation
        """
        for l in range(1, self.layers):
            self.w[l] = self.w[l] - alpha * dw[l]
            self.b[l] = self.b[l] - alpha * db[l]

    def test(self, x, y):
        """
        Computes accuracy and loss for a given set of x samples and corresponding labeled y classes
        :param x: Training data N x M matrix (N = feature vector dimension, M = # samples)
        :param y: Corresponding true y labels for data
        :return: Cost/loss and accuracy computed
        """
        z, a = self.forward(x)
        y_pred = a[self.layers - 1]
        cost = self.loss(y, y_pred)
        accuracy = accuracy_score(np.argmax(y, axis=0), np.argmax(y_pred, axis=0))
        return cost, accuracy

    def get_training_info_str(self, alpha, batch_size):
        """
        Returns a string label summarizing the training parameters
        """
        return u'g=%s, w_init=%s, layers=%s, \u03B1=%.2f, batch=%d' \
               % (self.activation.__name__, self.weight_init.__name__,
                  '-'.join([str(l) for l in self.layer_dims]), alpha, batch_size)

    def estimate_finite_diff_gradient(self, x_sample, y_sample, eps, weight_index):
        """
        Returns an approximation for the gradient using central finite difference derivative approximation
        dL/dw = [L(w+eps)-L(w-eps)]/2eps
        :param x_sample: X sample to test with
        :param y_sample: Y sample to test with
        :param eps: Epsilon value for central finite difference approximation
        :param weight_index: Coordinates of weight (layer_idx, neuron_idx, input_idx) to estimate using finite difference
        :return: Approximation for the gradient using central finite difference derivative approximation
        """
        (layer_idx, neuron_idx, input_idx) = weight_index
        weight_i = self.w[layer_idx][neuron_idx][input_idx]

        # Compute L(w-eps)
        self.w[layer_idx][neuron_idx][input_idx] = weight_i - eps
        z, a = self.forward(x_sample)
        train_cost_neg_eps = self.loss(y_sample, a[self.layers - 1])

        # Compute L(w+eps)
        self.w[layer_idx][neuron_idx][input_idx] = weight_i + eps
        z, a = self.forward(x_sample)
        train_cost_pos_eps = self.loss(y_sample, a[self.layers - 1])

        # Compute dL/dw = [L(w+eps)-L(w-eps)]/2eps
        self.w[layer_idx][neuron_idx][input_idx] = weight_i
        z, a = self.forward(x_sample)
        dw, db = self.backward(z, a, y_sample)
        return abs((train_cost_pos_eps - train_cost_neg_eps) / (2 * eps) - dw[layer_idx][neuron_idx][input_idx])

    def train(self, train_set, valid_set, alpha=0.1, epochs=10, batch_size=128, verbose=True):
        """
        Trains the neural network
        :param train_set: Training data set in (x,y) tuple format
        :param valid_set: Validation data set in (x,y) tuple format
        :param alpha: Learning rate to use during gradient descent
        :param epochs: Maximum number of epochs
        :param batch_size: Mini batch size
        :param verbose: If verbose logging is activated
        :return: Statistics from training: training set loss and accuracy, validation loss and accuracy
        """

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
        self.training_info_label = self.get_training_info_str(alpha, batch_size)
        if verbose:
            print("\nTRAINING: %s" % self.training_info_label)

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

        print("DONE after %ds: %s - ValidLoss=%f, ValidAcc=%f" % (time() - start, self.training_info_label,
                                                                                 valid_lost[-1], valid_acc[-1]))
        return train_lost, train_acc, valid_lost, valid_acc


class NNFactory(object):
    """
    Factory helper class for creating neural network models
    """
    DIGITS = 10
    MNIST_IMAGE_SIZE = 784

    @staticmethod
    def create(hidden_dims, activation=Sigmoid, weight_init=Glorot, in_dim=MNIST_IMAGE_SIZE, out_dim=DIGITS, rand_seed=1):
        """
        Creates a new neural network instance
        :param hidden_dims: Number of hidden dimensions
        :param activation: Activation function: [Sigmoid, Tanh, Relu]
        :param weight_init: Weight initialization scheme: [Zeros, Normal, Glorot]
        :param in_dim: Input dimension
        :param out_dim: Output dimension (number of classes)
        :param rand_seed: Optional seed for reproducibility of results. Used when initializing random weights
        :return: Created neural network model
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        layer_dims = [in_dim] + hidden_dims + [out_dim]
        return NN(layer_dims, activation, weight_init)
