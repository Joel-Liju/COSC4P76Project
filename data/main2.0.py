import math
import random
import sys

import numpy
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

    for i in range(len(data)):
        rand1 = random.randrange(0, len(data))
        rand2 = random.randrange(0, len(data))
        dtemp = data[rand1]
        ttemp = target[rand1]
        data[rand1] = data[rand2]
        target[rand1] = target[rand2]
        data[rand2] = dtemp
        target[rand2] = ttemp

    return Bunch(data=data, target=target)


def fill_param_array(start, end, by, labels_arr):
    arr = []
    while start < end:
        labels_arr.append(f'{start}')
        arr.append(math.pow(2, start))
        start += by
    return arr


def find_best_params(file_name, n_features, c_arr, g_arr, kernel='rbf', degree=3):

    c_ylabels = []
    gamma_xlabels = []
    c_vals = fill_param_array(c_arr[0], c_arr[1], c_arr[2], c_ylabels)
    gamma_vals = fill_param_array(g_arr[0], g_arr[1], g_arr[2], gamma_xlabels)


    hmap_array = [[] for i in range(len(c_vals))]

    accuracy_values_record = [[0 for g in gamma_vals] for c in c_vals]

    for run in range(10):
        data = loadData(file_name, n_features)
        for i_c, c in enumerate(c_vals):
            for i_g, g in enumerate(gamma_vals):
                clf = svm.SVC(kernel=kernel, gamma=g, C=c)
                scores = cross_val_score(clf, data.data, data.target, cv=10)
                mean = scores.mean()

                accuracy_values_record[i_c][i_g] += mean

    best_params = (0, 0, -1)
    for c in range(len(c_vals)):
        for g in range(len(gamma_vals)):
            m = (accuracy_values_record[c][g]/10.0) * 100
            hmap_array[c].append(m)
            if m > best_params[2]:
                best_params = (c_vals[c], gamma_vals[g], m)

    ax = sns.heatmap(hmap_array, linewidth=0.5, annot=True, vmin=47, vmax=54, xticklabels = gamma_xlabels, yticklabels = c_ylabels)
    ax.tick_params(axis='x', rotation=35)
    ax.set(xlabel='gamma', ylabel='C')
    ax.tick_params(axis='both', which='major', labelsize=7)

    print(m)
    print('---')
    return best_params

def find_best_params_median(file_name, n_features, c_arr, g_arr, kernel='rbf', degree=3):

    c_ylabels = []
    gamma_xlabels = []
    c_vals = fill_param_array(c_arr[0], c_arr[1], c_arr[2], c_ylabels)
    gamma_vals = fill_param_array(g_arr[0], g_arr[1], g_arr[2], gamma_xlabels)

    accuracy_values_record = []

    for run in range(10):
        best_params = (0, 0, -1)
        data = loadData(file_name, n_features)
        for i, c in enumerate(c_vals):
            for g in gamma_vals:
                clf = svm.SVC(kernel=kernel, gamma=g, C=c)
                scores = cross_val_score(clf, data.data, data.target, cv=10)
                mean = scores.mean()
                if mean > best_params[2]:
                    best_params = (c, g, mean)

        accuracy_values_record.append( (best_params[0], best_params[1], best_params[2])  )

    accuracy_values_record.sort(key=lambda x:x[2])
    median_values = accuracy_values_record[len(accuracy_values_record) // 2]
    best_params = (median_values[0], median_values[1], median_values[2])

    return best_params


def test(file, n_features, best_params_dic, kernel='rbf'):
    C, gamma, mean_acc = find_best_params_median(file, n_features, [-5, 15, 2], [-15, 3, 2], kernel)
    plt.clf()

    finer_c = math.log(C, 2)
    finer_gamma = math.log(gamma, 2)

    C, gamma, mean_acc = find_best_params(file,n_features, [finer_c-2, finer_c+2, 0.25], [finer_gamma-2, finer_gamma+2, 0.25], kernel)
    plt.savefig(f'./figures/{file[:-4]}_{kernel}.png')
    plt.clf()
    if best_params_dic[file][-1] < mean_acc:
        best_params_dic[file] = (C, gamma, kernel, mean_acc)

def latex_table(results):
    with open('latex_table.txt', 'a') as table:
        for k in results.keys():
            C, gamma, kernel, acc = results[k]
            table.write(f'{k}&{C}&{gamma}&{kernel}&{"%0.2f" % (acc)}\\\\')
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

    # test(files[1], 3, best_params, kernel='sigmoid')