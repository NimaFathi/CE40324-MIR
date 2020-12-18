import pickle
from collections import defaultdict


# for the bigram implementation the two pairs including the required term

class BiGramIndex:
    def __init__(self, name, docs, ids):
        self.name = name
        self.index = defaultdict(dict)

        self.construct_doc_list(docs, ids)

    def construct_doc_list(self, docs, ids):
        for doc_id, doc in zip(ids, docs):
            for token in doc:
                self.add_token(doc_id, token)

    def convert_to_bytestream(self, file):
        pickle.dump(self.index, file, pickle.HIGHEST_PROTOCOL)

    def add_token(self, doc_id, token):
        bounded_token = '$' + token + '$'
        for i in range(len(bounded_token) - 1):
            term = bounded_token[i: i + 2]

            if token not in self.index[term].keys():
                self.index[term][token] = []
            if doc_id not in self.index[term][token]:
                self.index[term][token].append(doc_id)

            if term not in self.index:
                self.index[term] = dict()
            if token not in self.index[term]:
                self.index[term][token] = []

    # save and load part

    def load(self):
        with open('index/' + self.name + '.pkl', 'rb') as f:
            self.index = pickle.load(f)

    def save(self):
        with open('index/' + self.name + '.pkl', 'wb') as f:
            self.convert_to_bytestream(f)

    #
    def delete_doc(self, doc_id):
        for term in self.index.keys():
            for token in self.index[term].keys():
                self.index[term][token].remove(doc_id)

    def show_bigram(self, bigram):
        print(list(self.index[bigram].keys()))
