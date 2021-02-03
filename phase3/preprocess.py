import collections
import nltk
import string
from nltk.stem import PorterStemmer
import hazm
import numpy as np

from sklearn.decomposition import PCA

from .clustering import load_data, vectorize


def pca(n_components, vectors, random_state=666):
    pca = PCA(n_components, random_state=random_state)
    return pca.fit_transform(vectors)


data, major_labels, minor_labels = load_data(stem=False, lemmatize=True, remove_conjunctions=True)
tf_idf, w2v = vectorize(data, w2v_options=dict(size=64, iter=128, min_count=2))
pca_tfidf = pca(64, tf_idf.toarray(), random_state=666)

