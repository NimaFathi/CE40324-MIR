from .base_classifier import BaseClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNClassifier(BaseClassifier):
    def __init__(self, k, x_train, y_train):
        super().__init__(x_train, y_train, classifier=None)
        self.k = k
        self.standard_scaler = StandardScaler()
        self.x = None
        self.y = None

    def fit(self):
        self.standard_scaler.fit(self.x_train)
        # use fit_transform on training data
        self.x = self.standard_scaler.fit_transform(self.x_train)
        self.y = self.y_train

    def get_neighbors(self, item):
        train = []
        for i in range(self.x.shape[0]):
            train.append((self.y[i], abs(np.linalg.norm(item - self.x[i], 2))))
        train.sort(key=lambda t: t[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append(train[i][0])
        return neighbors

    def predict(self, test):
        # use transform on test data
        x_test = self.standard_scaler.transform(test)
        y_predict = []
        for data in x_test:
            neighbors = self.get_neighbors(data)
            y_predict.append(max(set(neighbors), key=neighbors.count))
        return np.array(y_predict)
