

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score, v_measure_score, adjusted_rand_score, completeness_score
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import pandas as pd
import functools
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import itertools






def kmeans(data, vectors, n_components=None, **kwargs):
    n_components = len(data['major_cls'].unique()) if n_components is None else n_components
    model = KMeans(n_components, **kwargs)
    labels = model.fit_predict(vectors)
    return labels, None


def hierarchical(data, vectors, n_components=None, **kwargs):
    n_components = len(data['major_cls'].unique()) if n_components is None else n_components
    model = AgglomerativeClustering(n_components, **kwargs)
    labels = model.fit_predict(vectors)
    return labels, None


def GMM(data, vectors, n_components=None, **kwargs):
    n_components = len(data['major_cls'].unique()) if n_components is None else n_components
    model = GaussianMixture(n_components, **kwargs)
    model.fit(vectors)
    sizes = model.predict_proba(vectors)
    return model.predict(vectors), sizes