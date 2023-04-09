from DMC import DMC
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
from Utils import run_epochs_from
import numpy as np
import pandas as pd


def test_knn_iris(k):
    iris = load_iris()
    knn = KNN(k)

    print('\n\n---------- KNN for IRIS with {} neighbor ------------------\n'.format(k))

    return run_epochs_from(knn, iris.data, iris.target)


def test_dmc_iris():
    iris = load_iris()
    dmc = DMC()

    print('\n\n---------- DMC for IRIS ------------------\n')

    return run_epochs_from(dmc, iris.data, iris.target)


test_knn_iris(1)
test_dmc_iris()