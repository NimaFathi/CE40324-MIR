"""
this code brought here from first phase
"""

import pickle
import os
from collections import defaultdict

# for positional indexing a single word has a list of documents


class PositionalIndex:
    def __init__(self, name, docs, ids):
        self.index = defaultdict(dict)
        self.name = name

        # index is the dictionary of differnet terms

        self.construct_doc_list(docs, ids)

    def construct_doc_list(self, docs, ids):
        for doc_id, doc_info in zip(ids, docs):
            self.add_doc(doc_id, doc_info)

    # operations for making the posting lists dynamic

    def delete_doc(self, doc_id):
        for term in self.index.keys():
            self.index[term].pop(doc_id, None)

    def add_doc(self, doc_id, doc):
        for position, term in enumerate(doc):

            if doc_id not in self.index[term]:
                self.index[term][doc_id] = []
            self.index[term][doc_id].append(position + 2)

            if term not in self.index:
                self.index[term] = dict()

    # for part 2.2 the constructed posting list will be shown as bellow

    def show_posting_list(self, term):
        print(self.name + ":")
        for doc_id, positions in self.index[term].items():
            print(str(doc_id) + " : --> the positions are  " + str(positions))

    def show_position(self, term, doc_id):
        print(self.name + " positional index")
        print(self.index[term][doc_id])

