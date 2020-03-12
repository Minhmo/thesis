import json
import os

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import Utils.dataTools
import numpy as np

dataDir = 'authorData'  # Data directory
dataFilename = 'authorshipData.mat'  # Data filename
dataPath = os.path.join(dataDir, dataFilename)  # Data path

ratio_train = 0.6
ratio_valid = 0.2
ratio_test = 0.2

N_DATA_SPLITS = 10

doPrint = True
all_author_names = ['abbott', 'stevenson', 'alcott', 'alger', 'allen', 'austen', 'bronte', 'cooper', 'dickens',
                    'garland', 'hawthorne', 'james', 'melville', 'page', 'thoreau', 'twain', 'doyle', 'irving', 'poe',
                    'jewett', 'wharton']


def train_knn(k, data):
    result = []

    for split_n in range(N_DATA_SPLITS):
        data.get_split_same_author()

        X, y = data.getSamples('train')
        X = preprocessing.scale(X)

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, y)

        X_valid, y_val = data.getSamples('valid')
        X_valid = preprocessing.scale(X_valid)

        y_hat = neigh.predict(X_valid)

        totalErrors = np.sum(np.abs(y_hat - y_val) > 1e-9)
        accuracy = 1 - totalErrors.item() / len(y_val)

        result.append(accuracy)

    return result


def train_svm(data):
    result = []

    for split_n in range(N_DATA_SPLITS):
        data.get_split_same_author()

        X, y = data.getSamples('train')
        X = preprocessing.scale(X)

        svc = SVC()
        svc.fit(X, y)

        X_valid, y_val = data.getSamples('valid')
        X_valid = preprocessing.scale(X_valid)

        y_hat = svc.predict(X_valid)

        totalErrors = np.sum(np.abs(y_hat - y_val) > 1e-9)
        accuracy = 1 - totalErrors.item() / len(y_val)

        result.append(accuracy)

    return result


def train_networks():
    K = [3, 5, 8, 11, 13, 17]
    knn_search_results = {}
    svc_search_results = {}
    for author in all_author_names:
        data = Utils.dataTools.Authorship(author, ratio_train, ratio_valid, dataPath)

        knn_search_results[author] = {}

        # train KNN
        for k in K:
            result = train_knn(k, data)
            knn_search_results[author][k] = result

        # train SVM
        svc_search_results[author] = train_svm(data)
    # mean_svc = np.mean(svc_search_results)
    # means_knn = [np.mean(v) for k, v in knn_search_results.items()]
    with open('knn_results.txt', 'w+') as outfile:
        json.dump(knn_search_results, outfile)
    with open('svm_results.txt', 'w+') as outfile:
        json.dump(svc_search_results, outfile)


def analyze_results():
    knn_file = open('results/knn_results.txt', 'r')
    svm_file = open('results/svm_results.txt', 'r')

    knn_search_results = json.load(knn_file)
    svm_search_results = json.load(svm_file)

    knn_results = {}
    svm_results = {}

    for author in all_author_names:
        temp = knn_search_results[author]

        means_knn = [np.mean(v) for k, v in temp.items()]
        max_acc = np.max(means_knn)

        knn_results[author] = max_acc
        svm_results[author] = np.mean(svm_search_results[author])

    knn_file.close()
    svm_file.close()

    return svm_results, knn_results


if __name__ == '__main__':
    # data = Utils.dataTools.Authorship('poe', ratio_train, ratio_valid, dataPath)
    # train_networks()
    svm_results, knn_results = analyze_results()
