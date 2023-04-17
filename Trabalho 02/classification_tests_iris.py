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

    print('\n\n---------- KNN para dados IRIS com {} vizinho ------------------\n'.format(k))

    return run_epochs_from(knn, iris.data, iris.target)


def test_dmc_iris():
    iris = load_iris()
    dmc = DMC()

    print('\n\n---------- DMC para base IRIS ------------------\n')

    return run_epochs_from(dmc, iris.data, iris.target)


def plot_correlation_iris():
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    feature_correlation = iris_df.corr()

    print('\n--- Mapa de calor para correlação das features da base de dados IRIS ---')
    sns.heatmap(feature_correlation, annot=True, cmap='coolwarm')
    plt.show()

    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df['class'] = iris['target']
    correlations = df.corr()['class'][:-1]
    correlations_df = pd.DataFrame(correlations)
    correlations_df.columns = ['Correlação com as saidas']
    correlations_df.index = iris['feature_names']

    sns.heatmap(correlations_df, annot=True, cmap='coolwarm')
    plt.title('\nMapa de calor das correlações entre as features e a classe na base de dados IRIS')
    plt.show()

    print('\n ------ Scatter plot dos atributos petal width (cm) e petal length (cm)------ ')

    petal_length = iris_df['petal length (cm)']
    petal_width = iris_df['petal width (cm)']

    plt.figure(figsize=(8, 6))
    plt.scatter(petal_length, petal_width, c=iris.target)

    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')

    plt.show()


def test_knn_with_two_features_iris(k):
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)[['petal length (cm)', 'petal width (cm)']]
    knn = KNN(k)

    print('\n\n---------- KNN para dados IRIS com uso de duas features: petal length (cm) e petal width (cm), '
          'com KNN com {} vizinho  ------------------\n'.format(k))

    return run_epochs_from(knn, iris_df, iris.target)


def test_dmc_with_two_features_iris():
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)[['petal length (cm)', 'petal width (cm)']]
    dmc = DMC()

    print('\n\n---------- DMC para Iris com duas features: petal length (cm) e petal width (cm)------------------\n')

    return run_epochs_from(dmc, iris_df, iris.target)


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

