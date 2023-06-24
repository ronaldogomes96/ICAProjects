import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
import seaborn as sns


def load_torneamento_database():
    df = pd.read_csv('torneamento.csv', header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def run_classification_epochs_from(model, X, y, epochs=50):
    X = X.values if isinstance(X, pd.DataFrame) else X

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    accuracy_results_train = []
    accuracy_results_tests = []
    list_of_matrix_tests = []

    for epoch in range(epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        model.fit(X_train, y_train)

        y_predict_train = np.array(model.predict(X_train))
        y_predict_list = np.array(model.predict(X_test))

        list_of_matrix_tests.append(confusion_matrix(y_test, y_predict_list))

        accuracy_results_train.append(accuracy_score(y_train, y_predict_train))
        accuracy_results_tests.append(accuracy_score(y_test, y_predict_list))

    show_confusion_matrix(list_of_matrix_tests)

    accuracy_mean_train = np.mean(accuracy_results_train)
    std_train = np.std(accuracy_results_train)
    accuracy_mean_test = np.mean(accuracy_results_tests)
    std_test = np.std(accuracy_results_tests)

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def show_confusion_matrix(list_of_matrix_tests):
    sum_confusion_matrix_test = list_of_matrix_tests[0]

    for i in range(1, 50):
        sum_confusion_matrix_test += list_of_matrix_tests[i]

    mean_confusion_matrix_test = sum_confusion_matrix_test / 50

    sns.heatmap(mean_confusion_matrix_test, annot=True, cmap='Blues')
    plt.title('Matriz de Confusão da Média das 50 Rodadas para fase Testes')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Valores Verdadeiros')
    plt.show()


def choose_number_of_neurons_for_mlp(X, y):
    mean = select_number_neurons_mlp_mean_method(X, y)
    mlp_mean = run_classification_epochs_from(MLPClassifier(hidden_layer_sizes=(mean,),
                                                            activation='logistic',
                                                            max_iter=500), X, y)

    sqtr = select_number_neurons_mlp_sqrt_method(X, y)
    mlp_sqtr = run_classification_epochs_from(MLPClassifier(hidden_layer_sizes=(sqtr,),
                                                            activation='logistic',
                                                            max_iter=500), X, y)

    kolmogorov = select_number_neurons_mlp_komogorov_method(X)
    mlp_kolmogorov = run_classification_epochs_from(MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                                                                  activation='logistic',
                                                                  max_iter=500), X, y)

    mean_and_std_features = {
        'Média da acuracia MLP da fase de treinamento': [mlp_mean[0] * 100, mlp_sqtr[0] * 100,
                                                         mlp_kolmogorov[0] * 100],
        'Média da acuracia MLP da fase de testes': [mlp_mean[2] * 100, mlp_sqtr[2] * 100,
                                                    mlp_kolmogorov[2] * 100],
        'Desvio padrão da MLP fase de treino': [mlp_mean[1] * 100, mlp_sqtr[1] * 100, mlp_kolmogorov[1] * 100],
        'Desvio padrão da MLP fase de testes': [mlp_mean[3] * 100, mlp_sqtr[3] * 100, mlp_kolmogorov[3] * 100]
    }
    line_names = ['Valor medio - {} neuronios'.format(mean),
                  'Raiz quadrada - {} neuronios'.format(sqtr),
                  'KOMOGOROV - {} neuronios'.format(kolmogorov)]

    results = pd.DataFrame(mean_and_std_features, index=line_names)

    return results


def select_number_neurons_mlp_mean_method(X, y, is_spine=False):
    y_trasnfor = y.shape[1] if is_spine else len(set(y))
    return int((X.shape[1] + y_trasnfor) / 2)


def select_number_neurons_mlp_sqrt_method(X, y, is_spine=False):
    y_trasnfor = y.shape[1] if is_spine else len(set(y))
    return int(np.sqrt(X.shape[1] * y_trasnfor))


def select_number_neurons_mlp_komogorov_method(X):
    return int(2 * X.shape[1] + 1)


def plot_score_for_choose_epochs(epochs, X, y, scaler, number_of_neurons):
    X = scaler.fit_transform(X)

    mlp = MLPClassifier(hidden_layer_sizes=(number_of_neurons,),
                        activation='logistic',
                        max_iter=epochs)

    history = mlp.fit(X, y).loss_curve_

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history, label='Treino')
    ax.set_title('Curva de Perda', fontsize=28)
    ax.set_xlabel('Épocas', fontsize=24)
    ax.set_ylabel('Perda', fontsize=24)

    ax.set_xlim(0, len(history))

    ax.legend()

    plt.show()

