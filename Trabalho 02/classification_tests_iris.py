from DMC import DMC
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
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


def plot_correlation_iris():
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    feature_correlation = iris_df.corr()

    print('\n--- Correlation between Iris features ---')
    print(feature_correlation)

    print('\n--- Heat map from correlation features ---')
    sns.heatmap(feature_correlation, annot=True, cmap='coolwarm')
    plt.show()

    print('\n The two features more correlated are: petal length (cm) and petal width (cm) \n')

    print('\n ------ Scatter plot of the attributes ------ ')

    petal_length = iris_df['petal length (cm)']
    petal_width = iris_df['petal width (cm)']

    plt.figure(figsize=(8, 6))
    plt.scatter(petal_length, petal_width, c=iris.target)

    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')

    plt.show()

