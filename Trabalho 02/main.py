from DMC import DMC
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.datasets import load_iris
from Utils import show_confusion_matrix
import numpy as np
import pandas as pd


def test_knn_iris(k):
    iris = load_iris()

    knn = KNN(k)

    accuracy_results = []

    print('\n\n---------- KNN for IRIS with {} neighbor ------------------\n'.format(k))

    for epoch in range(50):
        print('\n\nEpoch {}'.format(epoch + 1))
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=None)

        knn.train(X_train, y_train)

        y_predict_list = []

        for index in range(X_test.shape[0]):
            y_predict_list.append(knn.predict(X_test[index]))

        print('\nConfusion matrix')
        show_confusion_matrix(y_test, y_predict_list)

        accuracy = accuracy_score(y_test, y_predict_list)
        print('\nAcurracy: {} %'.format(accuracy*100))
        accuracy_results.append(accuracy)

    return accuracy_results


test_knn_iris(1)





