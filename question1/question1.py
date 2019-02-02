from time import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from weight_initialization import Normal, Glorot, Zeros
from activations import Sigmoid, Tanh, Relu
from models import NN
from preprocessing import load_mnist, ParamsSearchResults
import matplotlib.pyplot as plt
import pandas as pd
import threading


def weight_initialization_tests():
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
        plt.savefig('results/weight_init_%s.png' % nn.weight_init.__name__)
        plt.show()


def parameter_search_worker(x_train, y_train, x_valid, y_valid, params, params_search_results):
    for (g, h, a, b) in params:
        layer_config = [x_train.shape[0]] + h + [y_train.shape[0]]
        nn = NN(layer_config, activation=g)
        nn.train(x_train, y_train, x_valid, y_valid, alpha=a, batch_size=b, verbose=False)
        cost, accuracy = nn.test(x_valid, y_valid)
        params_search_results.insert(nn, a, b, accuracy)


def parameter_search():
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
    threads = [threading.Thread(target=parameter_search_worker, args=(x_train, y_train, x_valid, y_valid, p, params_search_results)) for p in param_chunks]
    [t.start() for t in threads]
    [t.join() for t in threads]
    params_search_results.display()

    # for (g, h, a, b) in params:
    #     layer_config = [x_train.shape[0]] + h + [y_train.shape[0]]
    #     nn = NN(layer_config, activation=g)
    #     nn.train(x_train, y_train, x_valid, y_valid, alpha=a, batch_size=b, verbose=False)
    #     cost, accuracy = nn.test(x_valid, y_valid)
    #     params_search_results.insert(nn, a, b, accuracy)

    # params_search_results.display()


def debug_gradients():
    pass


if __name__ == '__main__':
    # weight_initialization_tests()
    parameter_search()
    # np.random.seed(1)
    #
    # # nn = NN(X_train.shape[0], [512, 256], y_train.shape[0], weight_init=Normal)
    # # start = time()
    # # nn.train(X_train, y_train, X_test, y_test, epochs=10, alpha=0.1)
    # # print(time() - start)
    # #
    # # nn = NN(X_train.shape[0], [512, 256], y_train.shape[0], weight_init=Zeros)
    # # start = time()
    # # nn.train(X_train, y_train, X_test, y_test, epochs=10, alpha=0.1)
    # # print(time() - start)
    #
    # nn = NN(X_train.shape[0], [512, 256], y_train.shape[0], weight_init=Glorot)
    # start = time()
    # nn.train(X_train, y_train, X_test, y_test, epochs=10, alpha=1, batch_size=64)
    # print(time() - start)
    #
    # cost, y_pred = nn.test(X_test, y_test)
    # predictions = np.argmax(y_pred, axis=0)
    # labels = np.argmax(y_test, axis=0)
    # print(classification_report(labels, predictions))
    # print(accuracy_score(labels, predictions))
