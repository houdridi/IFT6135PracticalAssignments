"""
IFT 6135 - W2019 - Practical Assignment 1 - Question 1
Assignment Instructions: https://www.overleaf.com/read/msxwmbbvfxrd
Github Repository: https://github.com/stefanwapnick/IFT6135PracticalAssignments
"""
import threading

import matplotlib.pyplot as plt
import numpy as np

from activations import Sigmoid, Tanh, Relu
from models import NN
from preprocessing import load_mnist, ParamsSearchResults
from weight_initialization import Normal, Glorot, Zeros
from time import time


def weight_initialization_test():
    # Fixed random seed for reproducibility
    np.random.seed(1)
    x_train, y_train, x_valid, y_valid = load_mnist()
    layer_dims = [x_train.shape[0]] + [512, 256] + [y_train.shape[0]]
    alpha = 0.1
    batch_size = 256

    models = [
        NN(layer_dims, weight_init=Zeros),
        NN(layer_dims, weight_init=Normal),
        NN(layer_dims, weight_init=Glorot)
    ]

    for nn in models:
        train_cost, validation_cost = nn.train(x_train, y_train, x_valid, y_valid, alpha=alpha, batch_size=batch_size)
        plt.plot(np.arange(len(train_cost)) + 1, train_cost, label='Training Set')
        plt.plot(np.arange(len(train_cost)) + 1, validation_cost, label='Validation Set')
        plt.title(nn.get_training_info_str(alpha, batch_size))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        # plt.savefig('results/weight_init_%s.png' % nn.weight_init.__name__)
        plt.show()


def parameter_search_worker(x_train, y_train, x_valid, y_valid, params, params_search_results):
    for (g, h, a, b) in params:
        layer_config = [x_train.shape[0]] + h + [y_train.shape[0]]
        nn = NN(layer_config, activation=g)
        nn.train(x_train, y_train, x_valid, y_valid, alpha=a, batch_size=b, verbose=False)
        cost, accuracy = nn.test(x_valid, y_valid)
        params_search_results.insert(nn, a, b, accuracy)


def parameter_search_test():
    # Fixed random seed for reproducibility
    np.random.seed(1)
    x_train, y_train, x_valid, y_valid = load_mnist()

    params_search_results = ParamsSearchResults.load()
    activations = [Sigmoid, Tanh, Relu]
    alphas = [0.1, 0.01]
    batch_sizes = [128, 256]
    hidden_layers = [[512, 256], [256, 512], [512, 512]]
    n_threads = 4

    params = [(g, h, a, b)
              for g in activations for a in alphas
              for b in batch_sizes for h in hidden_layers]

    param_chunks = [params[i::n_threads] for i in xrange(n_threads)]
    threads = [threading.Thread(target=parameter_search_worker, args=(x_train, y_train, x_valid, y_valid, p,
                                                                      params_search_results)) for p in param_chunks]
    [t.start() for t in threads]
    start = time()
    [t.join() for t in threads]
    params_search_results.display()
    print("Parameter search done after %ds" % (time()-start))


def finite_difference_gradient_test():
    # Fixed random seed for reproducibility
    np.random.seed(1)

    x_train, y_train, x_valid, y_valid = load_mnist()

    layer_config = [x_train.shape[0]] + [512, 256] + [y_train.shape[0]]
    nn = NN(layer_config, activation=Sigmoid)
    nn.train(x_train, y_train, x_valid, y_valid)

    m = 10
    idx = np.squeeze(np.argwhere(x_train[:, 0] > 0)[:m])
    x_sample = x_train[:, 0].reshape((-1, 1))
    y_sample = y_train[:, 0].reshape((-1, 1))
    n = 10. ** (np.arange(5))
    epsilons = np.reciprocal(n)
    error = np.zeros(len(epsilons))

    for i_eps, eps in enumerate(epsilons):
        for i in idx:
            gradient_error = nn.debug_gradient(x_sample, y_sample, eps, (1, 0, i))
            error[i_eps] = max(error[i_eps], gradient_error)

    plt.semilogy(n, error)
    plt.title('Max Error in Gradient For First 10 Weights vs. N')
    plt.xlabel("N")
    plt.ylabel("Error")
    # plt.savefig('results/gradient_difference.png')
    plt.show()
