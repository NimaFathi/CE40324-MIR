import numpy as np


def idf_(position_indexes, dictionary, document_ids):
    vec = np.array([len(position_indexes[token]) for token in dictionary], dtype=np.float)
    non_zero = vec != 0
    N = len(document_ids)
    try:
        vec[non_zero] = np.log10(N / vec[non_zero])
    except ZeroDivisionError:
        print('there is no documents in IR system!')
    return vec


def vector_tnt(position_indexes, dictionary, document_ids):
    doc_vectors = []
    idf = idf_(position_indexes, dictionary, document_ids)
    for doc_id in document_ids:
        tf = np.array(
            [len(position_indexes.get(token, {}).get(doc_id, [])) for token in dictionary], dtype=np.float
        )
        doc_vectors.append(tf * idf)
    return doc_vectors
