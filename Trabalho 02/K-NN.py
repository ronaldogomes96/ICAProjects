import numpy as np
import pandas as pd


class KNN:
    def __init__(self, k):
        self.y = None
        self.X = None
        self.k = k

    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        X = X.values
        list_of_distances = []

        for index in range(self.X.shape[0]):
            # Euclidean distance calculation
            square = np.square(self.X - X)
            sum = np.sum(square.loc[index])

            list_of_distances.append(np.sqrt(sum))

        '''
            argsort returns the indices that sort a NumPy array in ascending order. 
            In other words, it returns the indexes which, if used to access the array elements, 
            would result in a sorted version of the array.
        '''
        k_indexes = np.argsort(list_of_distances)[:self.k]

        classes = self.y[k_indexes]

        '''
            bincount counts the number of occurrences of each non-negative integer value in a NumPy array. 
            It returns an array with the count of occurrences of each non-negative integer value, 
            where the array index corresponds to the integer value.
        '''
        counts = np.bincount(classes)

        '''
            argmax returns the index of the element that contains the maximum value.
        '''
        return np.argmax(counts)
