from sklearn import svm
from sklearn.model_selection import cross_val_score
import random
from sklearn.utils import Bunch
import numpy as np
import csv

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

files = {'complete.csv': ['sigmoid', 724.077,1.189], 'complete2019.csv': ['rbf', 1722.156,0.5], 'complete2020.csv': ['sigmoid',215.270, 3.364], 'NoTrudeau.csv': ['rbf',2048.0,0.421], 'NoTrudeau2019.csv': ['rbf',1024.0, 2.0],
         'NoTrudeau2020.csv': ['sigmoid',128.0, 6.727], 'NoTrump.csv': ['rbf',362.039, 1.414], 'NoTrump2019.csv': ['rbf', 22.627, 4.0], 'NoTrump2020.csv': ['linear', 0.0078125, 7.629e-06], 'onlyMoving.csv': ['rbf',0.0078125, 7.629e-06],
         'onlyMoving2019.csv': ['rbf', 0.0078125, 7.629e-06], 'onlyMoving2020.csv': ['rbf',0.0078125, 7.629e-06]}

runs = 30

with open('stats.csv', 'a') as stats_file:
    stats_file.write('file,run,accuracy\n')

    for i, f in enumerate(files.keys()):
        if i < 3:
            features = 3
        elif i < 9:
            features = 2
        else:
            features = 1
        for r in range(runs):
            data = loadData(f, features)
            clf = svm.SVC(kernel=files[f][0], gamma=files[f][1], C=files[f][2])
            scores = cross_val_score(clf, data.data, data.target, cv=10)
            mean = scores.mean()
            stats_file.write(f'{f},{r},{mean}\n')

