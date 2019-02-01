from time import time

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from models import NN
from preprocessing import load_mnist

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()

    np.random.seed(138)

    nn = NN(X_train.shape[0], [64, 64], y_train.shape[0])
    start = time()
    nn.train(X_train, y_train, X_test, y_test, epochs=10)
    print(time() - start)

    cost, y_pred = nn.test(X_test, y_test)
    predictions = np.argmax(y_pred, axis=0)
    labels = np.argmax(y_test, axis=0)
    print(classification_report(labels, predictions))
    print(accuracy_score(labels, predictions))
