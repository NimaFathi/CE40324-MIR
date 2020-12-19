from .base_classifier import BaseClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNClassifier(BaseClassifier):
    def __init__(self, k, x_train, y_train):
        super().__init__(x_train, y_train, classifier=KNNClassifier(k, x_train, y_train))
        self.k = k
        self.standard_scaler = StandardScaler()

    def fit(self):
        self.standard_scaler.fit(self.x_train)
        self.x_train = self.standard_scaler.transform(self.x_train)

    @staticmethod
    def majority(lst):
        return max(set(lst), key=lst.count)

    @staticmethod
    def distance(v1, v2):
        return abs(np.linalg.norm(v2 - v1, 2))

    def neighbors(self, x):
        train = []
        for i in range(self.x_train.shape[0]):
            train.append((self.y_train[i], self.distance(self.x_train[i], x)))
        train.sort(key=lambda t: t[1])
        return [train[i][0] for i in range(self.k)]

    def predict(self, test):
        x_test = self.standard_scaler.transform(test)
        y_pred = []
        for data in x_test:
            y_pred.append(self.majority(self.neighbors(data)))
        return np.array(y_pred)
