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
import matplotlib.pyplot as plt
import numpy as np

import os


def plot_training_stats(stats, plot_title, plot_acc=False, save_as_file=None):
    """
    Plots training statistics: training loss and accuracy, validation loss and accuracy
    :param stats: Statistics to plot
    :param plot_title: Title of plot
    :param plot_acc: If accuracy should be plotted in addition to loss
    :param save_as_file: Optional file name to save results under
    """
    train_loss, train_acc, valid_loss, valid_acc = stats
    plt.plot(np.arange(len(train_loss)) + 1, train_loss, label='Training Set')
    plt.plot(np.arange(len(valid_loss)) + 1, valid_loss, label='Validation Set')
    plt.title('Loss Vs. Epochs\n%s' % plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    if save_as_file:
        plt.savefig(os.path.join('results', 'loss_%s' % save_as_file))
    plt.show()

    if not plot_acc:
        return

    plt.plot(np.arange(len(train_acc)) + 1, train_acc, label='Training Set')
    plt.plot(np.arange(len(valid_acc)) + 1, valid_acc, label='Validation Set')
    plt.title('Accuracy Vs. Epochs\n%s' % plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    if save_as_file:
        plt.savefig(os.path.join('results', 'acc_%s' % save_as_file))
    plt.show()


def plot_gradient_difference(N, error, save_as_file=None):
    """
    Plots error between gradient computed using back-propagation and that estimated with the central
    finite difference approximation
    :param N: N parameter used to compute epsilon using in the central finite difference approximation. epsilon = 1/N
    :param error: Error between gradient computed using back-propagation and that estimated using finite difference
    :param save_as_file: Optimal file to save results under
    """
    plt.semilogy(N, error)
    plt.title('Max Gradient Error For First 10 Weights of 2nd Layer vs. N')
    plt.xlabel("N")
    plt.ylabel("Error")
    if save_as_file:
        plt.savefig(os.path.join('results', save_as_file))
    plt.show()

