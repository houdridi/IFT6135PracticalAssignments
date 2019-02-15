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
import threading

import matplotlib.pyplot as plt
import numpy as np

from activations import Sigmoid, Tanh, Relu
from models import NN
from data import load_mnist, ResultsCache
from weight_initialization import Normal, Glorot, Zeros
from time import time
import os


def run_and_plot_results(nn, train_set, valid_set, alpha, batch_size, file_name=None):
    train_loss, train_acc, valid_loss, valid_acc = nn.train(train_set, valid_set, alpha=alpha, batch_size=batch_size)

    plt.plot(np.arange(len(train_loss)) + 1, train_loss, label='Training Set')
    plt.plot(np.arange(len(valid_loss)) + 1, valid_loss, label='Validation Set')
    plt.title('Loss Vs. Epochs\n%s' % nn.get_training_info_str(alpha, batch_size))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    if file_name:
        plt.savefig(os.path.join('results', 'loss_%s' % file_name))
    plt.show()

    plt.plot(np.arange(len(train_acc)) + 1, train_acc, label='Training Set')
    plt.plot(np.arange(len(valid_acc)) + 1, valid_acc, label='Validation Set')
    plt.title('Accuracy Vs. Epochs\n%s' % nn.get_training_info_str(alpha, batch_size))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    if file_name:
        plt.savefig(os.path.join('results', 'acc_%s' % file_name))
    plt.show()


def weight_initialization_test():
    # Fixed random seed for reproducibility
    np.random.seed(1)
    (x_train, y_train), valid_set, _ = load_mnist()
    layer_dims = [x_train.shape[0]] + [512, 256] + [y_train.shape[0]]
    alpha = 0.1
    batch_size = 256

    models = [
        NN(layer_dims, weight_init=Zeros),
        NN(layer_dims, weight_init=Normal),
        NN(layer_dims, weight_init=Glorot)
    ]

    for nn in models:
        file_name = 'weight_init_{}.png'.format(nn.weight_init.__name__)
        run_and_plot_results(nn, (x_train, y_train), valid_set, alpha, batch_size, file_name=file_name)


def parameter_search_worker(train_set, valid_set, params, params_search_results):
    x_train, y_train = train_set
    for (g, h, a, b) in params:
        layer_config = [x_train.shape[0]] + h + [y_train.shape[0]]
        nn = NN(layer_config, activation=g)
        _, _, _, valid_acc = nn.train(train_set, valid_set, alpha=a, batch_size=b, verbose=False)
        params_search_results.insert(nn, a, b, valid_acc[-1])


def parameter_search_test(n_threads=1):
    # Fixed random seed for reproducibility
    np.random.seed(1)
    train_set, valid_set, _ = load_mnist()

    params_search_results = ResultsCache.load()
    activations = [Relu]
    alphas = [0.1]
    batch_sizes = [128]
    hidden_layers = [[512, 1024]]

    params = [(g, h, a, b)
              for g in activations for a in alphas
              for b in batch_sizes for h in hidden_layers]

    if n_threads > 1:
        param_chunks = [params[i::n_threads] for i in range(n_threads + 1)]
        threads = [threading.Thread(target=parameter_search_worker, args=(train_set, valid_set, p,
                                                                          params_search_results)) for p in param_chunks]
        [t.start() for t in threads]
        start = time()
        [t.join() for t in threads]
        print("Parameter search done after %ds" % (time() - start))
    else:
        parameter_search_worker(train_set, valid_set, params, params_search_results)

    params_search_results.display()


def finite_difference_gradient_test(save_figure=False):
    # Fixed random seed for reproducibility
    np.random.seed(1)

    (x_train, y_train), valid_set, _ = load_mnist()

    layer_config = [x_train.shape[0]] + [512, 256] + [y_train.shape[0]]
    nn = NN(layer_config, activation=Sigmoid)
    nn.train((x_train, y_train), valid_set)

    layer = 2
    M = 10
    N = 10. ** (np.arange(5))
    epsilons = np.reciprocal(N)
    error = np.zeros(len(epsilons))

    x_sample = x_train[:, 0].reshape((-1, 1))
    y_sample = y_train[:, 0].reshape((-1, 1))

    for i_eps, eps in enumerate(epsilons):
        for idx in range(M):
            gradient_error = nn.estimate_finite_diff_gradient(x_sample, y_sample, eps, (layer, 0, idx))
            error[i_eps] = max(error[i_eps], gradient_error)

    plt.semilogy(N, error)
    plt.title('Max Gradient Error For First 10 Weights of 2nd Layer vs. N')
    plt.xlabel("N")
    plt.ylabel("Error")
    if save_figure:
        plt.savefig('results/validate_gradient.png')
    plt.show()

