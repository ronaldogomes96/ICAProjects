from DMC import DMC
from KNN import KNN
from sklearn.datasets import load_wine
import seaborn as sns
import matplotlib.pyplot as plt
from Utils import run_epochs_from
import numpy as np
import pandas as pd


def test_knn_wine(k):
    wine = load_wine()
    knn = KNN(k)

    print('\n\n---------- KNN for Wine with {} neighbor ------------------\n'.format(k))

    return run_epochs_from(knn, wine.data, wine.target)


def test_dmc_wine():
    wine = load_wine()
    dmc = DMC()

    print('\n\n---------- DMC for Wine ------------------\n')

    return run_epochs_from(dmc, wine.data, wine.target)

