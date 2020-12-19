from sklearn.ensemble import RandomForestClassifier
from .base_classifier import BaseClassifier


class RFClassifier(BaseClassifier):
    def __init__(self, x_train, y_train):
        super(RFClassifier, self).__init__(
            x_train=x_train, y_train=y_train, classifier=RandomForestClassifier()
        )

