from .base_classifier import BaseClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler


class KNNClassifier(BaseClassifier):
    def __init__(self, k, x_train, y_train):
        super().__init__(x_train, y_train, classifier=None)
        self.k = k
        self.standard_scaler = StandardScaler()

    def fit(self):
        self.standard_scaler.fit(self.x_train)
        self.x_train = self.standard_scaler.transform(self.x_train)

    def neighbors(self, x):
        train = []
        for i in range(self.x_train.shape[0]):
            train.append((self.y_train[i], abs(np.linalg.norm(x - self.x_train[i], 2))))
        train.sort(key=lambda t: t[1])
        return [train[i][0] for i in range(self.k)]

    def predict(self, test):
        x_test = self.standard_scaler.transform(test)
        y_pred = []
        for data in x_test:
            y_pred.append(max(set(self.neighbors(data)), key=self.neighbors(data).count))
        return np.array(y_pred)
