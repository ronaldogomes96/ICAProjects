from Utils import load_torneamento_database, run_classification_epochs_from, \
    choose_number_of_neurons_for_mlp, plot_score_for_choose_epochs
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_mlp_for_chose_number_of_neurons():
    X, y = load_torneamento_database()
    results = choose_number_of_neurons_for_mlp(X, y)
    return results


def run_mlp_plot_score_for_epochs(epochs=8000):
    X, y = load_torneamento_database()

    kolmogorov = int(2 * X.shape[1] + 1)

    scaler = StandardScaler()

    plot_score_for_choose_epochs(epochs, X, y, scaler, kolmogorov)


def run_mlp_torneamento(epochs=5000):
    X, y = load_torneamento_database()

    kolmogorov = int(2 * X.shape[1] + 1)

    mlp = MLPClassifier(hidden_layer_sizes=(kolmogorov,),
                        activation='logistic',
                        max_iter=epochs)

    accuracy_mean_train, std_train, accuracy_mean_test, std_test = run_classification_epochs_from(mlp,
                                                                                                  X, y)
    return accuracy_mean_train, std_train, accuracy_mean_test, std_test
