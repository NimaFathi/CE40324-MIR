import matplotlib.pyplot as plt
import pandas as pd
import functools
import itertools
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE
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
# unindent?

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
            f'outputs/{cluster_type.__name__}-tfidf.csv')
        result[['link', 'w2v']].rename(columns={'link': 'link', 'w2v': 'pred'}).to_csv(
            f'outputs/{cluster_type.__name__}-w2v.csv')

    return result


def sk_tools(true_labels, predicted_labels):
    matrix = contingency_matrix(true_labels, predicted_labels)
    return {
        'purity': np.sum(np.amax(matrix, axis=0)) / np.sum(matrix),
        'adjusted_mutual_info': adjusted_mutual_info_score(true_labels, predicted_labels),
        'adjusted_rand_index': adjusted_rand_score(true_labels, predicted_labels),
    }


def get_res(kmeans_res=None, gmm_res=None, hier_res=None, data=None):
    res = defaultdict(list)
    for name, value in [('kmeans', kmeans_res), ('gmm', gmm_res), ('hierarchical', hier_res)]:
        if value is None:
            continue
        for vectorization in ['tf-idf', 'w2v']:
            res['algorithm'].append(name.capitalize())
            res['vectorization'].append(vectorization)
            alres = sk_tools(data['major_cls'], value[vectorization])
            for metric, metric_value in alres.items():
                res[metric].append(metric_value)
    return pd.DataFrame(res)


def grid_search(algorithm, data, tfidf=None, w2v=None, fixed_params=None, variables=None):
    vectors = []
    if tfidf is not None:
        vectors.append(('tf-idf', tfidf))
    if w2v is not None:
        vectors.append(('w2v', w2v))

    variable_keys = list(variables.keys())
    variables = variables or dict()
    fixed_params = fixed_params or dict()
    ans = defaultdict(list)
    for values in tqdm(list(itertools.product(*[variables[key] for key in variable_keys]))):
        cur_vars = dict()
        for i, key in enumerate(variable_keys):
            cur_vars[key] = values[i]

        for vec_name, vec in vectors:
            try:
                labels, sizes = algorithm(data, vec, **fixed_params, **cur_vars)
                result = get_res(data['major_cls'], labels)
                for met, met_val in result.items():
                    for var, var_val in cur_vars.items():
                        ans[var].append(var_val)
                    ans['metric'].append(met)
                    ans['score'].append(met_val)
                    ans['vectorization'].append(vec_name)
            except Exception:
                print("ERROR OCCURRED... ANS NOT UPDATED!\n")
    return pd.DataFrame(ans)


def plot2d(vectors, labels, true_labels=None, sizes=None, title=None):
    n_components = 2
    pca = PCA(n_components, random_state=666)
    vector = pca.fit_transform(vectors)
    vector_tsne = TSNE(n_components=n_components).fit_transform(vectors)

    if sizes is not None:
        sizes = sizes - sizes.min()
        sizes = (sizes / sizes.max()) * 40 + 10
    if true_labels is not None:
        fig, axes = plt.subplots(1, 4, figsize=(26, 5))
        axes[0].scatter(vector[:, 0], vector[:, 1], c=labels, s=sizes)
        axes[0].set_title('Prediction (PCA)')

        axes[1].scatter(vector_tsne[:, 0], vector_tsne[:, 1], c=labels, s=sizes)
        axes[1].set_title('Prediction (TSNE)')

        axes[2].scatter(vector[:, 0], vector[:, 1], c=true_labels)
        axes[2].set_title('Ground truth (PCA)')

        axes[3].scatter(vector_tsne[:, 0], vector_tsne[:, 1], c=true_labels)
        axes[3].set_title('Ground truth (TSNE)')
        if title:
            fig.suptitle(title)
        return

    plt.scatter(vector[:, 0], vector[:, 1], c=labels, s=sizes)
    if title:
        plt.title(title)
    plt.grid()
    plt.show()


def load_data(file='./files/hamshahri.json', stem=False, lemmatize=True, remove_conjunctions=False, join=' '):
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
