import numpy as np


class TfIdfSearch:
    def __init__(self, document_ids, position_indexes):
        self.document_ids = document_ids
        self.position_indexes = position_indexes

        self.dictionary = list(self.position_indexes.keys())

    def answers(self, query, no_wanted_outcomes):
        query_vector = self.query_vector(query=query)
        scores = []
        for doc_id in self.document_ids:
            scores.append((self.tf_idf_score(query_vector, doc_id)))
        ranked_docs = sorted(scores, key=lambda cell: cell[0], reverse=True)
        return ranked_docs[:min(no_wanted_outcomes, len(ranked_docs))]

    def tf_idf_score(self, query_vector, doc_id):
        doc_vector = self.document_vector(doc_id)
        return query_vector.dot(doc_vector), doc_id

    def idf(self):
        vec = np.array([len(self.position_indexes[token]) for token in self.dictionary], dtype=np.float)
        non_zero = vec != 0
        N = len(self.document_ids)
        try:
            vec[non_zero] = np.log10(N / vec[non_zero])
        except ZeroDivisionError:
            print('there is no documents in IR system!')
        return vec

    def tf_log_document(self, document_id):
        vector = np.array(
            [len(self.position_indexes[token].get(document_id, [])) for token in self.dictionary], dtype=np.float
        )
        # find non zero cells for get log
        nonzero_vec = vector != 0
        vector[nonzero_vec] = 1 + np.log(vector[nonzero_vec])
        return vector

    def document_vector(self, document_id):
        idf = self.idf()
        tf = self.tf_log_document(document_id)
        return self.norm(tf * idf)

    def tf_log_query(self, query):
        vector = np.zeros(len(self.dictionary))
        for token in query:
            if token in self.dictionary:
                vector[self.dictionary.index(token)] += 1

        # find non zero cells for get log
        nonzero_vec = vector != 0
        vector[nonzero_vec] = 1 + np.log(vector[nonzero_vec])
        return vector

    def query_vector(self, query):
        return self.norm(self.tf_log_query(query))

    def norm(self, vector):
        norm = np.sqrt(np.dot(vector, vector))
        return vector / norm if norm != 0 else vector
