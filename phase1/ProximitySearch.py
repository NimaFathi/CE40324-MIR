import itertools

from .tf_idf import TfIdfSearch


def query_find(query, id_, positional_index):
    dictionary = list(positional_index.keys())
    for term in query:
        if term in dictionary:
            if id_ not in positional_index[term]:
                return False
    return True


def proximity_search(ids_, query, positional_index, window):
    dictionary = list(positional_index.keys())
    col = []
    for id_ in ids_:
        if query_find(query, id_, positional_index):
            p_list = []
            for term in query:
                if term in dictionary:
                    p_list.append(positional_index[term][id_])
            d_list = list(itertools.product(*p_list))
            for item in d_list:
                dist = max(item) - min(item)
                if dist < window:
                    col.append(id_)
    return TfIdfSearch(col, positional_index).answers(query, no_wanted_outcomes=4)
