import math
import random
import sys

from sklearn import svm
import csv
from sklearn.utils import Bunch
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.ticker as ticker
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


def fill_param_array(start, end, by, labels_arr):
    arr = []
    while start < end:
        labels_arr.append(f'{start}')
        arr.append(math.pow(2, start))
        start += by
    return arr


def find_best_params(data, file_name, c_arr, g_arr, kernel='rbf', degree=3):
    c_ylabels = []
    gamma_xlabels = []
    c_vals = fill_param_array(c_arr[0], c_arr[1], c_arr[2], c_ylabels)
    gamma_vals = fill_param_array(g_arr[0], g_arr[1], g_arr[2], gamma_xlabels)


    hmap_array = [[] for i in range(len(c_vals))]

    best_params = (0, 0, -1)

    for i, c in enumerate(c_vals):
        for g in gamma_vals:
            clf = svm.SVC(kernel=kernel, gamma=g, C=c)
            scores = cross_val_score(clf, data.data, data.target, cv=10)
            mean = scores.mean()

            hmap_array[i].append(mean * 100)
            if mean > best_params[2]:
                best_params = (c, g, mean)


    # ax = sns.heatmap(hmap_array, linewidth=0.5, annot=True, vmin=44, vmax=58, xticklabels = gamma_xlabels, yticklabels = c_ylabels)
    # ax.tick_params(axis='x', rotation=35)
    # ax.set(xlabel='gamma', ylabel='C')
    # ax.tick_params(axis='both', which='major', labelsize=7)

    # def myticks(x, pos):
    #     return r"$2^{{ {} }}$".format(x)
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

    # plt.show()

    print(best_params[2])
    print('---')
    return best_params


def test(file, n_features, best_params_dic, seed=35, kernel='rbf', degree=3):

    data = loadData(file, n_features)
    C, gamma, mean_acc = find_best_params(data, file[:-4], [-5, 15, 2], [-15, 3, 2], kernel, degree)
    plt.clf()

    finer_c = math.log(C, 2)
    finer_gamma = math.log(gamma, 2)

    C, gamma, mean_acc = find_best_params(data, file[:-4], [finer_c-2, finer_c+2, 0.25], [finer_gamma-2, finer_gamma+2, 0.25], kernel, degree)
    # plt.savefig(f'./figures/{file[:-4]}_{kernel}.png')
    # plt.clf()
    if best_params_dic[file][-1] < mean_acc:
        best_params_dic[file] = (C, gamma, kernel, mean_acc)



    # X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3,
    #                                                     random_state=seed)  # 70% training and 30% test
    # clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("Final accuracy:", metrics.accuracy_score(y_test, y_pred))

    # clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)
    # scores = cross_val_score(clf, data.data, data.target, cv=10)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

def latex_table(results):
    with open('latex_table.txt', 'a') as table:
        for k in results.keys():
            C, gamma, kernel, acc = results[k]
            table.write(f'{k}&{C}&{gamma}&{kernel}&{"%0.2f" % (acc * 100)}\\\\')
            table.write('\n \\hline')


if __name__ == '__main__':
    files = ['complete.csv', 'complete2019.csv', 'complete2020.csv', 'NoTrudeau.csv', 'NoTrudeau2019.csv',
             'NoTrudeau2020.csv', 'NoTrump.csv', 'NoTrump2019.csv', 'NoTrump2020.csv', 'onlyMoving.csv',
             'onlyMoving2019.csv', 'onlyMoving2020.csv']

    best_params = {}
    for f in files:
        best_params[f] = (0, 0, "", 0)

    kernels = ['rbf', 'linear', 'sigmoid']
    for kernel in kernels:
        for i, f in enumerate(files):
            if i < 3:
                features = 3
            elif i < 9:
                features = 2
            else:
                features = 1
            test(f, features, best_params, kernel=kernel)
    latex_table(best_params)

    # test(files[7], 2, kernel='linear')