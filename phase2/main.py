import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .preproccessor import PreProcessEnglish
from .positional_index import PositionalIndex
from .tf_idf import TfIdfSearch


class Classifier:
    def __init__(self, file, processor: PreProcessEnglish):
        self.file_name = file
        self.processor = processor

    def read_data(self,):
        doc = pd.read_csv(self.file_name)
        processed_docs = self.processor.clean_documents(doc['description'], with_stop_words=False)
        processed_titles = self.processor.clean_documents(doc['title'], with_stop_words=True)
        doc_ids = list(range(len(processed_docs)))
        y = np.array(doc['views'])
        positional_index = PositionalIndex(name=self.file_name, docs=processed_docs, ids=doc_ids)
        positional_index.construct_doc_list(processed_titles, doc_ids)
        return doc_ids, y, positional_index

    def tf_idf_form(self, dictionary, doc_list: list, positional_index: dict, idf):
        doc_vectors = []
        for doc in doc_list:
            tf = np.array([len(positional_index.get(token, {}).get(doc, [])) for token in dictionary], dtype=np.float)
            doc_vectors.append(tf * idf)
        return doc_vectors

    # بخش سوم فاز دوم
    def final_evaluation(self, classifier_name, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = self.accuracy(tn, fp, fn, tp)
        precision_1, precision_2 = self.precision(tn, fp, fn, tp)
        recall_1, recall_2 = self.recall(tn, fp, fn, tp)
        class_1_f1, class_2_f1 = self.f1(precision_1, recall_1, precision_2, recall_2)
        print('evaluation of {}'.format(classifier_name))
        print('Accuracy is:{}'.format(accuracy))
        print('******************************')
        print('F1 with beta=1 & alpha=2: ')
        print('first class F1: {}', format(class_1_f1))
        print('second_class F1: {}'.format(class_2_f1))
        print('******************************')
        print('Precision, Recall first class: {}'.format(precision_1, recall_1))
        print('Precision, Recall second class: {}'.format(precision_2, recall_2))

    @staticmethod
    def accuracy(true_negative, false_positive, false_negative, true_positive):
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
        return accuracy

    @staticmethod
    def precision(true_negative, false_positive, false_negative, true_positive):
        class_1_precision = true_positive / (true_positive + false_positive)
        class_2_precision = true_negative / (true_negative + false_negative)
        print('first class precision: {}'.format(round(class_1_precision, 4)))
        print('seccond class precision: {}'.format(round(class_2_precision,4)))
        return class_1_precision, class_2_precision

    @staticmethod
    def recall(true_negative, false_positive, false_negative, true_positive):
        class_1_recall = true_positive / (true_positive + false_negative)
        class_2_recall = true_negative / (true_negative + false_positive)
        return class_1_recall, class_2_recall

    @staticmethod
    def f1(precision_1, recall_1, precision_2, recall_2):
        class_1_f1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        class_2_f1 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2)
        return class_1_f1, class_2_f1

