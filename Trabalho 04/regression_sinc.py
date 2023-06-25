import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_sinc():
    X_train = pd.read_csv('treino_sinc_dan.txt', sep='\s+', header=None, skipinitialspace=True,
                    engine='python')
    X_test = pd.read_csv('teste_sinc_dan.txt', sep='\s+', header=None, skipinitialspace=True,
                          engine='python')
    y_train = pd.read_csv('alvote_sinc_dan.txt', sep='\s+', header=None)
    y_test = pd.read_csv('alvote_sinc_dan.txt', sep='\s+', header=None)

    return X_train, X_test, y_train, y_test


def run_sin_regression(epochs):
    X_train, X_test, y_train, y_test = load_sinc()

    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_regression_epoch(X_train, X_test, y_train, y_test, number_of_neurons=90, epochs=epochs)

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_regression_epoch(X_train, X_test, y_train, y_test, number_of_neurons=3, epochs=10):
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    mse_results_train = []
    mse_results_tests = []

    y_predict_train_mean = []
    y_predict_list_mean = []

    for epoch in range(50):
        regressor = MLPRegressor(hidden_layer_sizes=(number_of_neurons,), max_iter=epochs)

        regressor.fit(X_train, y_train)

        y_predict_train = np.array(regressor.predict(X_train))
        y_predict_list = np.array(regressor.predict(X_test))

        y_predict_train_mean.append(y_predict_train)
        y_predict_list_mean.append(y_predict_list)

        mse_results_train.append(mean_squared_error(y_train, y_predict_train))
        mse_results_tests.append(mean_squared_error(y_test, y_predict_list))

    plt_results_train(y_train, np.mean(y_predict_train_mean, axis=0))
    plt_results_test(y_test, np.mean(y_predict_list_mean, axis=0))

    accuracy_mean_train = np.mean(mse_results_train)
    std_train = np.std(mse_results_train)
    accuracy_mean_test = np.mean(mse_results_tests)
    std_test = np.std(mse_results_tests)

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_mlp_for_chose_number_of_neurons():
    X_train, X_test, y_train, y_test = load_sinc()
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = [], [], [], []
    for neuron in range(2, 100):
        accuracy_mean_train_neuron, std_train_neuron, accuracy_mean_test_neuron, std_test_neuron = run_regression_epoch(X_train, X_test, y_train, y_test)
        accuracy_mean_train.append(accuracy_mean_train_neuron)
        std_train.append(std_train_neuron)
        accuracy_mean_test.append(accuracy_mean_test_neuron)
        std_test.append(std_test_neuron)

    print(accuracy_mean_train)
    print(accuracy_mean_test)


def plot_score_for_choose_epochs(epochs, number_of_neurons):
    X, _, y, _ = load_sinc()

    regressor  = MLPRegressor(hidden_layer_sizes=(number_of_neurons,), max_iter=epochs)

    history = regressor.fit(X, y).loss_curve_

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history, label='Treino')
    ax.set_title('Curva de Perda', fontsize=28)
    ax.set_xlabel('Épocas', fontsize=24)
    ax.set_ylabel('Perda', fontsize=24)

    ax.set_xlim(0, len(history))

    ax.legend()

    plt.show()


def plt_results_test(y_real, y_pred):
    # Criando uma figura e um eixo
    fig, ax = plt.subplots()

    # Plotando os resultados preditos e reais
    ax.plot(y_real, label='Real')
    ax.plot(y_pred, label='Predito')

    # Adicionando título e rótulos dos eixos
    ax.set_title("Resultado Predito vs Real")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Valor")
    ax.legend()

    # Exibindo o gráfico
    plt.show()


def plt_results_train(y_real, y_pred):
    # Criando uma figura e um eixo
    fig, ax = plt.subplots()

    # Plotando os resultados preditos e reais
    ax.plot(y_real, label='Real')
    ax.plot(y_pred, label='Predito')

    # Adicionando título e rótulos dos eixos
    ax.set_title("Resultado Predito vs Real")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Valor")
    ax.legend()

    # Exibindo o gráfico
    plt.show()


