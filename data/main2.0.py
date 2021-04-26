import math
import random
import sys

from sklearn import svm
import csv
from sklearn.utils import Bunch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import seaborn as sns
import matplotlib.pylab as plt


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


def fill_param_array(start, end, by):
    arr = []
    while start < end:
        arr.append(math.pow(2, start))
        start += by
    return arr


def find_best_params(data, file_name, c_arr, g_arr, kernel='rbf', degree=3):
    c_vals = fill_param_array(c_arr[0], c_arr[1], c_arr[2])
    gamma_vals = fill_param_array(g_arr[0], g_arr[1], g_arr[2])

    # exp = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
    # c_ylabels = [f'2{"-"if i < 0 else ""}{"" if (i//10)<1 else exp[(i//10)%10]}{exp[i%10]}' for i in range(-5, 17, 2)]
    # gamma_xlabels = [f'2{"-"if i < 0 else ""}{"" if (i//10)<1 else exp[(i//10)%10]}{exp[i%10]}' for i in range(-15, 5, 2)]

    # c_ylabels = [f'{i}' for i in range(-5, 17, 2)]
    # gamma_xlabels = [f'{i}' for i in range(-15, 5, 2)]

    hmap_array = [[] for i in range(len(c_vals))]

    best_params = ((), -1)

    for i, c in enumerate(c_vals):
        for g in gamma_vals:
            clf = svm.SVC(kernel=kernel, gamma=g, C=c)
            scores = cross_val_score(clf, data.data, data.target, cv=10)
            mean = scores.mean()

            hmap_array[i].append(mean * 100)
            if mean > best_params[1]:
                best_params = ((c, g), mean)

    # , xticklabels = gamma_xlabels, yticklabels = c_ylabels

    ax = sns.heatmap(hmap_array, linewidth=0.5, annot=True, vmin=40, vmax=58)
    ax.set(xlabel='gamma', ylabel='C')

    plt.show()
    # plt.savefig(f'./figures/{file_name}_{kernel}.png')
    # plt.clf()

    print(best_params[0])
    print(best_params[1])
    print('---')
    return best_params[0]


def test(file, n_features, seed=35, kernel='rbf', degree=3):
    data = loadData(file, n_features)
    C, gamma = find_best_params(data, file[:-4], [-5, 15, 2], [-15, 3, 2], kernel, degree)
    plt.clf()

    finer_c = math.log(C, 2)
    finer_gamma = math.log(gamma, 2)

    find_best_params(data, file[:-4], [finer_c-2, finer_c+2, 0.25], [finer_gamma-2, finer_gamma+2, 0.25], kernel, degree)
    # plt.savefig(f'./figures/{file[:-4]}_{kernel}.png')
    plt.clf()


    # X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3,
    #                                                     random_state=seed)  # 70% training and 30% test
    # clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("Final accuracy:", metrics.accuracy_score(y_test, y_pred))

    # clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    # scores = cross_val_score(clf, data.data, data.target, cv=10)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


if __name__ == '__main__':
    files = ['complete.csv', 'complete2019.csv', 'complete2020.csv', 'NoTrudeau.csv', 'NoTrudeau2019.csv',
             'NoTrudeau2020.csv', 'NoTrump.csv', 'NoTrump2019.csv', 'NoTrump2020.csv', 'onlyMoving.csv',
             'onlyMoving2019.csv', 'onlyMoving2020.csv']

    kernels = ['rbf', 'linear', 'sigmoid']
    for kernel in kernels:
        for i, f in enumerate(files):
            if i < 3:
                features = 3
            elif i < 9:
                features = 2
            else:
                features = 1
            test(f, features, kernel=kernel)

    # test(files[-2], 1)