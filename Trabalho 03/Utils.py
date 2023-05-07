import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns


def load_database(database_name):
    X = pd.read_csv('{}_input.txt'.format(database_name), sep='\s+', header=None, skipinitialspace=True, engine='python')
    y = pd.read_csv('{}_target.txt'.format(database_name), sep='\s+', header=None)

    y = transform_targets(y)

    return X, y


def transform_targets(y):
    y_transformed = []

    for index in range(y.shape[0]):
        y_transformed.append(np.argmax(y.iloc[index]))

    return np.array(y_transformed)


def run_epochs_from(model, X, y, epochs=50, scaler_name='minMax'):
    X = X.values if isinstance(X, pd.DataFrame) else X

    if scaler_name == 'minMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X = scaler.fit_transform(X)

    accuracy_results_train = []
    accuracy_results_tests = []
    list_of_matrix_train = []
    list_of_matrix_tests = []

    for epoch in range(epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        model.fit(X_train, y_train)

        y_predict_train = np.array(model.predict(X_train))
        y_predict_list = np.array(model.predict(X_test))

        list_of_matrix_train.append(confusion_matrix(y_train, y_predict_train))
        list_of_matrix_tests.append(confusion_matrix(y_test, y_predict_list))

        accuracy_results_train.append(accuracy_score(y_train, y_predict_train))
        accuracy_results_tests.append(accuracy_score(y_test, y_predict_list))

    show_confusion_matrix(list_of_matrix_train, list_of_matrix_tests)

    accuracy_mean_train = np.mean(accuracy_results_train)
    std_train = np.std(accuracy_results_train)
    accuracy_mean_test = np.mean(accuracy_results_tests)
    std_test = np.std(accuracy_results_tests)

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def show_confusion_matrix(list_of_matrix_train, list_of_matrix_tests):
    sum_confusion_matrix_train = list_of_matrix_train[0]
    sum_confusion_matrix_test = list_of_matrix_tests[0]

    for i in range(1, 50):
        sum_confusion_matrix_train += list_of_matrix_train[i]
        sum_confusion_matrix_test += list_of_matrix_tests[i]

    mean_confusion_matrix_train = sum_confusion_matrix_train / 50
    mean_confusion_matrix_test = sum_confusion_matrix_test / 50

    sns.heatmap(mean_confusion_matrix_train, annot=True, cmap='Blues')
    plt.title('Matriz de Confusão da Média das 50 Rodadas para fase Treinamento')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Verdadeiros')
    plt.show()

    sns.heatmap(mean_confusion_matrix_test, annot=True, cmap='Blues')
    plt.title('Matriz de Confusão da Média das 50 Rodadas para fase Testes')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Verdadeiros')
    plt.show()
