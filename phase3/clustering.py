import matplotlib.pyplot as plt
import pandas as pd
import functools
import itertools
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


from .preprocess import preprocessed_terms


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

def return_clustered_csv(data, cluster_type, tfidf=None, w2v=None, options=None, save=False):
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
            f'phase3/outputs/{cluster_type.__name__}-tfidf.csv')
        result[['link', 'w2v']].rename(columns={'link': 'link', 'w2v': 'pred'}).to_csv(
            f'phase3/outputs/{cluster_type.__name__}-w2v.csv')

    return result


def return_clustered_csv_gmm(data, cluster_type, tfidf=None, w2v=None, options=None, options_tfidf=None,
                             options_w2v=None, save=False):
    options = options or dict()
    options_tfidf = options_tfidf or dict()
    options_w2v = options_w2v or dict()
    result = pd.DataFrame({'link': data['link']})

    if tfidf is not None:
        args = {**options, **options_tfidf}
        name = f'{cluster_type.__name__.capitalize()}' + ('' if not options else (
                ' (' + ','.join(f'{i}={j}' for i, j in args.items()) + ')'))
        result['tf-idf'], sizes = cluster_type(data, tfidf, **args)
        plot2d(tfidf, result['tf-idf'], true_labels=data['major_cls'], sizes=sizes, title=f'{name} [tf-idf]')
    if w2v is not None:
        args = {**options, **options_w2v}
        name = f'{cluster_type.__name__.capitalize()}' + ('' if not options else (
                ' (' + ','.join(f'{i}={j}' for i, j in args.items()) + ')'))
        result['w2v'], sizes = cluster_type(data, w2v, **args)
        plot2d(w2v, result['w2v'], true_labels=data['major_cls'], sizes=sizes, title=f'{name} [w2v]')
    if save:
        result[['link', 'tf-idf']].rename(columns={'link': 'link', 'tf-idf': 'pred'}).to_csv(
            f'phase3/outputs/{cluster_type.__name__.lower()}-tfidf.csv')
        result[['link', 'w2v']].rename(columns={'link': 'link', 'w2v': 'pred'}).to_csv(
            f'phase3/outputs/{cluster_type.__name__.lower()}-w2v.csv')
    return result

def sk_tools(true_labels, predicted_labels):
    matrix = contingency_matrix(true_labels, predicted_labels)
    return {
        'purity': np.sum(np.amax(matrix, axis=0)) / np.sum(matrix),
        'mutual_info': adjusted_mutual_info_score(true_labels, predicted_labels),
        'rand_index': adjusted_rand_score(true_labels, predicted_labels),
    }


def get_res(kmeans_res=None, gmm_res=None, hier_res=None, data=None):
    ans = defaultdict(list)
    for name, value in [('kmeans', kmeans_res), ('gmm', gmm_res), ('hierarchical', hier_res)]:
        if value is not None:
            for vectorization in ['tf-idf', 'w2v']:
                ans['algorithm'].append(name.capitalize())
                ans['vectorization'].append(vectorization)
                alres = sk_tools(data['major_cls'], value[vectorization])
                for metric, metric_value in alres.items():
                    ans[metric].append(metric_value)
    return pd.DataFrame(ans)


def grid_search(algorithm, data, tfidf=None, w2v=None, fixed_params=None, variables=None):
    result = defaultdict(list)

    var_keys = list(variables.keys())
    fixed_params = fixed_params or dict()
    variables = variables or dict()

    vectors = []
    if tfidf is not None:
        vectors.append(('tf-idf', tfidf))
    if w2v is not None:
        vectors.append(('w2v', w2v))

    for vals in tqdm(list(itertools.product(*[variables[key] for key in var_keys]))):
        cur_vars = dict()
        for i, key in enumerate(var_keys):
            cur_vars[key] = vals[i]

        for vec_name, vec in vectors:
            try:
                labels, sizes = algorithm(data, vec, **fixed_params, **cur_vars)
                eval_res = sk_tools(data['major_cls'], labels)
                outer_func(result, vec_name, eval_res, cur_vars)
            except Exception:
                pass
    return pd.DataFrame(result)


def outer_func(result, vec_type, metrics, variables):
    for met, met_val in metrics.items():
        for var, var_val in variables.items():
            result[var].append(var_val)
        result['metric'].append(met)
        result['score'].append(met_val)
        result['vectorization'].append(vec_type)


def plot2d(vectors, labels, true_labels=None, sizes=None, title=None):
    n_components = 2
    pca = PCA(n_components, random_state=666)
    vector = pca.fit_transform(vectors)
    if sizes is not None:
        sizes = sizes - sizes.min()
        sizes = (sizes / sizes.max()) * 40 + 10
    if true_labels is not None:
        fig, axes = plt.subplots(1, 4, figsize=(26, 5))
        axes[0].scatter(vector[:, 0], vector[:, 1], c=labels, s=sizes)
        axes[0].set_title('Pred PCA')
        axes[1].scatter(vector[:, 0], vector[:, 1], c=true_labels)
        axes[1].set_title('True PCA')

        if title:
            fig.suptitle(title)
        return

    plt.scatter(vector[:, 0], vector[:, 1], c=labels, s=sizes)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def load_data(file='./phase3/hamshahri.json', stem=False, lemmatize=True, remove_conjunctions=False, join=' '):
    data = pd.read_json(file, encoding='utf-8')
    data['major_cls'], data['minor_cls'] = zip(*data['tags'].map(lambda x: tuple(x[0].split('>'))))
    major_labels, minor_labels = mapped_labels(data['major_cls']), mapped_labels(data['minor_cls'])
    data['major_cls'] = data['major_cls'].apply(lambda x: major_labels[x])
    data['minor_cls'] = data['minor_cls'].apply(lambda x: minor_labels[x])
    data['terms'] = (data['title'] + ' ' + data['summary']).apply(
        functools.partial(preprocessed_terms, stem=stem, lemmatize=lemmatize, remove_conjunctions=remove_conjunctions,
                          join=join))
    return data, major_labels, minor_labels


def mapped_labels(column):
    label_mapping = {label: i for i, label in enumerate(column.unique())}
    return label_mapping


# this part is for turning the tests into vectors using
# TFIDF and W2V from the gensim model library

def vectorize(data, w2v_options=None, tf_idf_options=None):
    n = 100
    w2v_options = w2v_options or dict(workers=8, iter=n)
    tf_idf_options = tf_idf_options or dict()
    vectorizer = TfidfVectorizer(**tf_idf_options)
    tf_idf = vectorizer.fit_transform(data['terms'])
    # split the data with their spaces and turn it into a vector

    model = Word2Vec(data['terms'].apply(lambda x: x.split(' ')), **w2v_options)
    w2v = np.array(data['terms'].apply(
        lambda x: sum(model.wv[y] if y in model.wv else 0 for y in x.split(' ')) / len(x.split(' '))).to_list())
    return tf_idf, w2v
