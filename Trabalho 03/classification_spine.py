from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Utils import load_database, run_epochs_from
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_perceptron_spine():
    X, y = load_database('coluna')

    perceptron = Perceptron(random_state=None)
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(perceptron, X, y, scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_mlp_simple_spine():
    X, y = load_database('coluna')

    mlp = MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        max_iter=30)
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(mlp, X, y, scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_mlp_for_chose_number_of_neurons():
    X, y = load_database('coluna')

    targets = pd.read_csv('{}_target.txt'.format('coluna'), sep='\s+', header=None)

    # Metodo do valor medio

    print('------- VALOR MEDIO --------')

    mean = int((X.shape[1] + targets.shape[1]) / 2)

    mlp_neurons_mean = MLPClassifier(hidden_layer_sizes=(mean,),
                                     activation='logistic',
                                     max_iter=1000)

    accuracy_mean_train, std_mean_train, accuracy_mean_test, std_mean_test = run_epochs_from(mlp_neurons_mean, X, y, scaler_name='standart')

    # Metodo da raiz quadrada

    print('------- RAIZ QUADRADA --------')

    sqtr = int(np.sqrt(X.shape[1] * targets.shape[1]))

    mlp_neurons_sqtr = MLPClassifier(hidden_layer_sizes=(sqtr,),
                                     activation='logistic',
                                     max_iter=1000)

    accuracy_sqtr_train, std_sqtr_train, accuracy_sqrt_test, std_sqtr_test = run_epochs_from(mlp_neurons_sqtr, X, y, scaler_name='standart')

    # KOLMOGOROV

    print('------- KOMOGOROV --------')

    kolmogorov = int(2 * X.shape[1] + 1)

    mlp_neurons_kolmogorov = MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                                           activation='logistic',
                                           max_iter=1000)

    accuracy_kolmogorov_train, std_kolmogorov_train, accuracy_kolmogorov_test, std_kolmogorov_test = run_epochs_from(
        mlp_neurons_kolmogorov, X, y, scaler_name='standart')

    mean_and_std_features = {
        'Média da acuracia MLP da fase de treinamento': [accuracy_mean_train * 100, accuracy_sqtr_train * 100,
                                                         accuracy_kolmogorov_train * 100],
        'Média da acuracia MLP da fase de testes': [accuracy_mean_test * 100, accuracy_sqrt_test * 100,
                                                    accuracy_kolmogorov_test * 100],
        'Desvio padrão da MLP fase de treino': [std_mean_train * 100, std_sqtr_train * 100, std_kolmogorov_train * 100],
        'Desvio padrão da MLP fase de testes': [std_mean_test * 100, std_sqtr_test * 100, std_kolmogorov_test * 100]
    }
    line_names = ['Valor medio - {} neuronios'.format(mean),
                  'Raiz quadrada - {} neuronios'.format(sqtr),
                  'KOMOGOROV - {} neuronios'.format(kolmogorov)]
    results = pd.DataFrame(mean_and_std_features, index=line_names)
    return results


def run_mlp_plot_score_for_epochs(epochs=10000):
    X, y = load_database('coluna')
    targets = pd.read_csv('{}_target.txt'.format('coluna'), sep='\s+', header=None)

    kolmogorov = int(2 * X.shape[1] + 1)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    mlp = MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                        activation='logistic',
                        max_iter=epochs)

    history = mlp.fit(X, y).loss_curve_

    # Crie uma figura e um eixo personalizados
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plote a curva de perda (loss) com todas as épocas
    ax.plot(history, label='train')

    # Defina o título e rótulos dos eixos
    ax.set_title('Curva de Perda', fontsize=18)
    ax.set_xlabel('Épocas', fontsize=14)
    ax.set_ylabel('Perda', fontsize=14)

    # Defina o limite do eixo x para mostrar todas as épocas
    ax.set_xlim(0, len(history))

    # Defina os valores do eixo x que serão mostrados
    xticks = [0, len(history) // 4, len(history) // 2, 3 * len(history) // 4, len(history) - 1]
    ax.set_xticks(xticks)

    # Adicione uma legenda
    ax.legend()

    # Mostra o gráfico
    plt.show()


def run_mlp_optimized():
    X, y = load_database('coluna')
    targets = pd.read_csv('{}_target.txt'.format('coluna'), sep='\t', header=None)
    kolmogorov = int(2 * X.shape[1] + 1)

    mlp = MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                        activation='logistic',
                        max_iter=1100)
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(mlp, X, y, scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test