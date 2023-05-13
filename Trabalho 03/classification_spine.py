from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Utils import load_database, run_epochs_from, choose_number_of_neurons_for_mlp, plot_score_for_choose_epochs
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_perceptron_spine():
    X, y = load_database('coluna')

    perceptron = Perceptron(random_state=None, tol=0.001)
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(perceptron, X, y,
                                                                                   scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_mlp_simple_spine():
    X, y = load_database('coluna')

    mlp = MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        max_iter=50)
    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(mlp, X, y,
                                                                                   scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test


def run_mlp_for_chose_number_of_neurons():
    X, y = load_database('coluna', target_trasnform=False)
    return choose_number_of_neurons_for_mlp(X, y, scaler_name='standart')


def run_mlp_plot_score_for_epochs(epochs=5000):
    X, y = load_database('coluna')

    kolmogorov = int(2 * X.shape[1] + 1)

    scaler = StandardScaler()

    plot_score_for_choose_epochs(epochs, X, y, scaler, kolmogorov)


def run_mlp_optimized(epochs):
    X, y = load_database('coluna')
    kolmogorov = int(2 * X.shape[1] + 1)

    mlp = MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                        activation='logistic',
                        max_iter=epochs)

    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_epochs_from(mlp, X, y, scaler_name='standart')

    return accuracy_mean_train, std_train, accuracy_mean_test, std_test
