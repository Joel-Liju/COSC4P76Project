import math
import random
from sklearn import svm
import csv
from sklearn.utils import Bunch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


def loadData(file, n_features):
    with open(file) as csv_file:
        data_file = csv.reader(csv_file)
        count = -1
        for line in data_file:
            count += 1

    with open(file) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = count
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(0 if sample[-1] == "DOWN" else 1, dtype=int)

    return Bunch(data=data, target=target)


def find_best_params(data, kernel='rbf'):
    c_vals = [math.pow(2, i) for i in range(-5, 17, 2)]
    gamma_vals = [math.pow(2, i) for i in range(-15, 5, 2)]

    best_params = ((), -1)

    accuracies = []

    for c in c_vals:
        for g in gamma_vals:
            clf = svm.SVC(kernel=kernel, gamma=g, C=c)
            scores = cross_val_score(clf, data.data, data.target, cv=10)
            mean = scores.mean()

            if "%0.2f" % (scores.mean()) not in accuracies:
                accuracies.append("%0.2f" % (scores.mean()))
            if mean > best_params[1]:
                best_params = ((c, g), mean)

    print(best_params[0])
    print(best_params[1])
    print(accuracies)
    return best_params[0]


def test(file, n_features, seed=35, kernel='rbf', degree=3):
    data = loadData(file, n_features)
    C, gamma = find_best_params(data, kernel)

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3,
                                                        random_state=seed)  # 70% training and 30% test
    clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Final accuracy:", metrics.accuracy_score(y_test, y_pred))

    # clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    # scores = cross_val_score(clf, data.data, data.target, cv=10)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))





