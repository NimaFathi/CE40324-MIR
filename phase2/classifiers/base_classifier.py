class BaseClassifier:
    def __init__(self, x_train, y_train, classifier):
        self.x_train = x_train
        self.y_train = y_train
        self.classifier = classifier

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.classifier.predict(X)
