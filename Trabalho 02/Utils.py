import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def euclidian_distance(x, y):
    square = np.square(x - y)
    sum = np.sum(square)

    return np.sqrt(sum)


def show_confusion_matrix(y_test, y_predict_list):
    matrix = confusion_matrix(y_test, y_predict_list)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    matrix_display.plot()
    plt.show()

