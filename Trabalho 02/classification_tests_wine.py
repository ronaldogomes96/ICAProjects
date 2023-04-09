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


def plot_correlation_wine():
    wine = load_wine()
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    feature_correlation = wine_df.corr()

    print('\n--- Correlation between Wine features ---')
    print(pd.DataFrame(data=feature_correlation, columns=wine.feature_names))

    print('\n--- Heat map from correlation features ---')
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=0.8)
    sns.heatmap(feature_correlation, annot=True, cmap='coolwarm')
    plt.show()

    print('\n The two features more correlated are: flavanoids and total_phenols \n')

    print('\n ------ Scatter plot of the attributes ------ ')

    flavanoids = wine_df['flavanoids']
    total_phenols = wine_df['total_phenols']

    plt.figure(figsize=(8, 6))
    plt.scatter(flavanoids, total_phenols, c=wine.target)

    plt.xlabel('flavanoids')
    plt.ylabel('total phenols')

    plt.show()


def test_knn_with_five_features_wine(k):
    wine = load_wine()
    principal_columns = ['flavanoids', 'total_phenols', 'od280/od315_of_diluted_wines', 'proanthocyanins', 'hue']
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)[principal_columns]
    knn = KNN(k)

    print('\n\n---------- KNN for Wine with five features and {} neighbor ------------------\n'.format(k))

    return run_epochs_from(knn, wine_df, wine.target)


def test_dmc_with_five_features_wine():
    wine = load_wine()
    principal_columns = ['flavanoids', 'total_phenols', 'od280/od315_of_diluted_wines', 'proanthocyanins', 'hue']
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)[principal_columns]
    dmc = DMC()

    print('\n\n---------- DMC for Wine with five features ------------------\n')

    return run_epochs_from(dmc, wine_df, wine.target)
