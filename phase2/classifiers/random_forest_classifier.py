from sklearn.ensemble import RandomForestClassifier
from .base_classifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    def __init__(self, x_train, y_train):
        super(RandomForestClassifier, self).__init__(
            x_train=x_train, y_train=y_train, classifier=RandomForestClassifier()
        )

