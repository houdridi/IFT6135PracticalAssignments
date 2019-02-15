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

from activations import Sigmoid, Tanh, Relu
from models import NN, NNFactory
from data import load_mnist, ResultsCache
from weight_initialization import Normal, Glorot, Zeros
from visualization import plot_gradient_difference, plot_training_stats
import numpy as np


def weight_initialization_test():
    train_set, valid_set, _ = load_mnist()
    hidden_layer_dims = [512, 256]
    alpha = 0.1
    batch_size = 256
    weight_inits = [Zeros, Normal, Glorot]

    for weight_init in weight_inits:
        nn = NNFactory.create(hidden_layer_dims, activation=Sigmoid, weight_init=weight_init)
        stats = nn.train(train_set, valid_set, alpha=alpha, batch_size=batch_size)
        plot_training_stats(stats, plot_title=nn.get_training_info_str(alpha, batch_size),
                            save_as_file='weight_init_{}.png'.format(nn.weight_init.__name__))


def parameter_search_test():
    # Parameters to search
    activations = [Sigmoid, Tanh, Relu]
    alphas = [0.1, 0.01]
    batch_sizes = [128, 256]
    hidden_layers = [[512, 256], [512, 512], [800, 512]]
    weight_inits = [Glorot]

    train_set, valid_set, _ = load_mnist()
    results_cache = ResultsCache.load()
    params = [(g, h, a, b, w)
              for g in activations for a in alphas for b in batch_sizes
              for h in hidden_layers for w in weight_inits]

    for (g, h, a, b, w) in params:
        nn = NNFactory.create(h, activation=g, weight_init=w)
        _, _, _, valid_acc = nn.train(train_set, valid_set, alpha=a, batch_size=b, verbose=False)
        results_cache.insert(nn, a, b, valid_acc[-1])
    results_cache.display()


def finite_difference_gradient_test():
    layer = 2
    M = 10
    N = 10. ** (np.arange(5))
    epsilons = np.reciprocal(N)
    error = np.zeros(len(epsilons))

    (x_train, y_train), valid_set, _ = load_mnist()
    nn = NNFactory.create(hidden_dims=[512, 256], activation=Sigmoid, weight_init=Glorot)
    nn.train((x_train, y_train), valid_set)

    x_sample = x_train[:, 0].reshape((-1, 1))
    y_sample = y_train[:, 0].reshape((-1, 1))

    for i_eps, eps in enumerate(epsilons):
        for idx in range(M):
            # weight idx = layer #, neuron #, weight # for neuron
            # Inspect 10 first weights of 2nd layer, 1st neuron
            weight_idx = (layer, 0, idx)
            gradient_error = nn.estimate_finite_diff_gradient(x_sample, y_sample, eps, weight_idx)
            error[i_eps] = max(error[i_eps], gradient_error)
    plot_gradient_difference(N, error, 'validate_gradient.png')


if __name__ == '__main__':
    # Part 1 - Build model basic test
    # train_set, valid_set, test_set = load_mnist()
    # model = NNFactory.create(hidden_dims=[512, 256], activation=Relu, weight_init=Glorot)
    # stats = model.train(train_set, valid_set, alpha=0.1, batch_size=128)
    # plot_training_stats(stats, )

    # Part 2 - Weight initialization
    weight_initialization_test()
    # Part 3 - hyperparameter search
    parameter_search_test()
    # Part 4 - Validate gradient using Finite difference
    finite_difference_gradient_test()
