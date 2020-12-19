from .base_classifier import BaseClassifier

import numpy as np
import math


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self, x_train, y_train):
        # HOLY SHIT THAT LOOKS SO WRONG (fixed, but i like this comment ngl)
        super().__init__(x_train, y_train, classifier=None)
        self.p_terms1 = None
        self.p_terms2 = None
        self.p_col1 = 0
        self.p_col2 = 0

    def fit(self):
        docs_in_col = self.x_train.shape[0]
        self.p_col1 = -sum(self.y_train[self.y_train < 0]) / docs_in_col
        self.p_col2 = 1 - self.p_col1

        doc_freq1 = []
        doc_freq2 = []
        terms_in_col1 = 0
        terms_in_col2 = 0
        for column in self.x_train.T:
            tc1 = sum(column[self.y_train < 0])
            tc2 = sum(column) - tc1
            doc_freq1.append(tc1 + 1)
            terms_in_col1 += tc1
            doc_freq2.append(tc2 + 1)
            terms_in_col2 += tc2

        distinct_terms_size = self.x_train.shape[1]
        prob_terms1 = []
        for freq in doc_freq1:
            prob_terms1.append(math.log2(freq / (distinct_terms_size + terms_in_col1)))
        self.p_terms1 = np.array(prob_terms1)
        prob_terms2 = []
        for freq in doc_freq2:
            prob_terms2.append(math.log2(freq / (distinct_terms_size + terms_in_col2)))
        self.p_terms2 = np.array(prob_terms2)

    def predict(self, test):
        labels = []
        for document in test:
            if not (math.log2(self.p_col1) + sum(self.p_terms1[document > 0]) <
                    math.log2(self.p_col2) + sum(self.p_terms2[document > 0])):
                labels.append(-1)
            else:
                labels.append(1)
        return np.array(labels)
