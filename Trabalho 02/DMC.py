import numpy as np
from Utils import euclidian_distance


class DMC:

    def __init__(self):
        self.centroids = None
        self.classes = None

    def train(self, df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        self.classes = np.unique(y)
        self.centroids = []

        for c in self.classes:
            X_c = X[y == c]
            centroid = np.mean(X_c, axis=0)
            self.centroids.append(centroid)

    def predict(self, X):
        list_of_distances = []

        for centroid_index in range(len(self.centroids)):
            # Euclidean distance calculation
            list_of_distances.append(euclidian_distance(self.centroids[centroid_index], X.values))

        '''
            argmin returns the index of the element that contains the minimum value.
        '''
        return self.classes[np.argmin(list_of_distances)]

