

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



# cluster type is one of the above three algorithm types 

 def return_clustered_csv(data, cluster_type, tfidf = None, w2v=None, options=None, save=False):

    options = options or dict()
    result = pd.DataFrame({'link': data['link']})
    name = f'{cluster_type.__name__}' + ('' if not options else (
            ' (' + ','.join(f'{i}={j}' for i, j in options.items()) + ')'))
    if tfidf is not None:
        result['tf-idf'], sizes = cluster_type(data, tfidf, **options)
        plot2d(tfidf, result['tf-idf'], true_labels=data['major_cls'], sizes=sizes, title=f'{name} [tf-idf]')

    if w2v is not None:
        result['w2v'], sizes = cluster_type(data, w2v, **options)
        plot2d(w2v, result['w2v'], true_labels=data['major_cls'], sizes=sizes, title=f'{name} [w2v]')

    if save:
        result[['link', 'tf-idf']].rename(columns={'link': 'link', 'tf-idf': 'pred'}).to_csv(
            f'outputs/{cluster_type.__name__}-tfidf.csv')
        result[['link', 'w2v']].rename(columns={'link': 'link', 'w2v': 'pred'}).to_csv(
            f'outputs/{cluster_type.__name__}-w2v.csv')
    return result   