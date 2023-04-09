import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def run_epochs_from(model, X, y, epochs=50):
    X = X.values if isinstance(X, pd.DataFrame) else X
    accuracy_results = []

    for epoch in range(epochs):
        print('\n\nEpoch {}'.format(epoch + 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

        model.train(X_train, y_train)

        y_predict_list = []

        for index in range(X_test.shape[0]):
            y_predict_list.append(model.predict(X_test[index]))

        print('\nConfusion matrix')
        show_confusion_matrix(y_test, y_predict_list)

        accuracy = accuracy_score(y_test, y_predict_list)
        print('\nAcurracy: {} %'.format(accuracy * 100))
        accuracy_results.append(accuracy)

    return accuracy_results


def euclidian_distance(x, y):
    square = np.square(x - y)
    sum = np.sum(square)

    return np.sqrt(sum)


def show_confusion_matrix(y_test, y_predict_list):
    matrix = confusion_matrix(y_test, y_predict_list)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    matrix_display.plot()
    plt.show()

