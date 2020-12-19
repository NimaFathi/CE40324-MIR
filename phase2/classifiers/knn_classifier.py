from .base_classifier import BaseClassifier

import numpy as np
import math
from sklearn.preprocessing import StandardScaler


class KNNClassifier(BaseClassifier):
    def __init__(self, k, x_train, y_train):
        super().__init__(x_train, y_train, classifier=None)
        self.k = k
        self.standard_scaler = StandardScaler()
        self.x = None
        self.y = None

    def fit(self):
        # self.standard_scaler.fit(self.x_train)
        # # use fit_transform on training data
        # self.x = self.standard_scaler.fit_transform(self.x_train)
        # self.y = self.y_train
        self.x = self.x_train
        self.y = self.y_train

    @staticmethod
    def euclidean_dist(p1, p2):
        dim, sum_ = len(p1), 0
        for index in range(dim - 1):
            sum_ += math.pow(p1[index] - p2[index], 2)
        return math.sqrt(sum_)

    def get_neighbors(self, item):
        train = []
        for i in range(self.x.shape[0]):
            train.append((self.y[i], self.euclidean_dist(item, self.x[i])))
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
