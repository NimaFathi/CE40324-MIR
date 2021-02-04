import string
import hazm

from sklearn.decomposition import PCA
from .clustering import load_data, vectorize

english_numerics = '0123456789'
persian_numerics = '\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9'
persian_puncts = '\u060C\u061B\u061F\u0640\u066A\u066B\u066C'

chars_to_remove = string.punctuation + persian_puncts + persian_numerics + english_numerics

persian_conjuction = {'از', 'به', 'با', 'بر', 'برای', 'در', 'و', 'که', 'را'}
persian_translator = str.maketrans('', '', chars_to_remove)

hazm_normalizer = hazm.Normalizer()
hazm_stemmer = hazm.Stemmer()
hazm_lemmatizer = hazm.Lemmatizer()


def preprocessed_terms(text, stem=False, lemmatize=False, remove_conjunctions=False, join=None):
    normalized_text = hazm_normalizer.normalize(text.translate(persian_translator))
    result = hazm.word_tokenize(normalized_text)
    if stem:
        result = [hazm_stemmer.stem(x) for x in result]
    if lemmatize:
        result = [hazm_lemmatizer.lemmatize(x) for x in result]
    if remove_conjunctions:
        result = [x for x in result if x not in persian_conjuction]
    if join is not None:
        return join.join(result)
    return result


def pca(n_components, vectors, random_state=666):
    return PCA(n_components, random_state=random_state).fit_transform(vectors)


data, major_labels, minor_labels = load_data(stem=False, lemmatize=True, remove_conjunctions=True)
tf_idf, w2v = vectorize(data, w2v_options=dict(size=64, iter=128, min_count=2))
pca_tfidf = pca(64, tf_idf.toarray(), random_state=666)
