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

    print('\n\n---------- KNN para dados Wine com {} vizinho ------------------\n'.format(k))

    return run_epochs_from(knn, wine.data, wine.target)


def test_dmc_wine():
    wine = load_wine()
    dmc = DMC()

    print('\n\n---------- DMC para base Wine ------------------\n')

    return run_epochs_from(dmc, wine.data, wine.target)


def plot_correlation_wine():
    wine = load_wine()
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

    df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    df['class'] = wine['target']
    correlations = df.corr()['class'][:-1]
    correlations_df = pd.DataFrame(correlations)
    correlations_df.columns = ['Correlação com as saidas']
    correlations_df.index = wine['feature_names']

    sns.heatmap(correlations_df, annot=True, cmap='coolwarm')
    plt.title('\nMapa de calor das correlações entre as features e a classe na base de dados Wine')
    plt.show()

    print('\n ------ Scatter plot dos atributos ------ ')

    flavanoids = wine_df['flavanoids']
    total_phenols = wine_df['od280/od315_of_diluted_wines']

    plt.figure(figsize=(8, 6))
    plt.scatter(flavanoids, total_phenols, c=wine.target)

    plt.xlabel('flavanoids')
    plt.ylabel('od280/od315_of_diluted_wines')

    plt.show()


def test_knn_with_five_features_wine(k):
    wine = load_wine()
    principal_columns = ['flavanoids', 'total_phenols', 'od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'nonflavanoid_phenols']
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)[principal_columns]
    knn = KNN(k)

    print('\n\n---------- KNN para base de dados Wine com cinco features e {} vizinhos ------------------\n'.format(k))

    return run_epochs_from(knn, wine_df, wine.target)


def test_dmc_with_five_features_wine():
    wine = load_wine()
    principal_columns = ['flavanoids', 'total_phenols', 'od280/od315_of_diluted_wines', 'alcalinity_of_ash', 'nonflavanoid_phenols']
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)[principal_columns]
    dmc = DMC()

    print('\n\n---------- DMC para base de dados Wine com cinco features ------------------\n')

    return run_epochs_from(dmc, wine_df, wine.target)


def show_accuracy_mean_and_confusion_matrix(accuracys, matrixs):
    print('\n\n---------- Média das acurácias: {} %'.format(np.mean(accuracys)*100))

    sum_cm = matrixs[0]
    for i in range(1, 50):
        sum_cm += matrixs[i]

    mean_cm = sum_cm / 50

    sns.heatmap(mean_cm, annot=True, cmap='Blues')
    plt.title('Matriz de Confusão da Média das 50 Rodadas')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Verdadeiros')
    plt.show()