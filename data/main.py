import random
from sklearn import svm
import csv
from sklearn.utils import Bunch
import numpy as np

def loadData(file,n_features):
    with open(file) as csv_file:
        data_file = csv.reader(csv_file)
        count = -1
        for line in data_file:
            count+=1
    
    with open(file) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = count
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(0 if sample[-1]=="DOWN" else 1, dtype=int)

    return Bunch(data=data, target=target)

def test(file,n_features,seed=35,kernel='linear',degree=3, gamma='scale',C=1):

    data = loadData(file,n_features)

    from sklearn.model_selection import train_test_split

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3,random_state=seed) # 70% training and 30% test
    clf = svm.SVC(kernel=kernel,degree=degree,gamma=gamma,C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics

    # Model Accuracy: how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    def k_fold(data_size,k=10): # returns division of training data
        remainder = data_size%k
        quotient = data_size//k
        indices =[]
        left = 0
        for i in range (0,k):
            right = left + quotient - 1
            if remainder > 0:
                right+=1
                remainder-=1
            indices.append((left,right))
            left = right + 1
        return indices

    def split_data(data, indices, seed=0): # this splits the data k fold
        random.seed(seed)
        random.shuffle(data)
        split = [data[s:e+1] for s,e in indices]
        return split
