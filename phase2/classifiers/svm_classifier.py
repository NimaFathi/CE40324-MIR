from sklearn.svm import SVC
from .base_classifier import BaseClassifier


class SoftMarginSVMClassifier(BaseClassifier):
    def __init__(self, x_train, y_train,  C):
        super(SoftMarginSVMClassifier, self).__init__(
            x_train=x_train, y_train=y_train, classifier=SVC(C=C)
        )
