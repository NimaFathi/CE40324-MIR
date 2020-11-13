from src.preprocess import PreProcess, PreProcessEnglish, PreProcessPersian
import pandas as pd
import xml.etree.ElementTree as ET
from src.PositionalIndex import PositionalIndex
from src.BigramIndex import BiGramIndex
from src.tf_idf import TfIdfSearch
import datetime


class IRSystem:
    TYPE_ENGLISH = 'english'
    TYPE_PERSIAN = 'persian'
    TYPES = [TYPE_ENGLISH, TYPE_PERSIAN]

    def __init__(self, ir_type):
        self.type = ir_type
        self.pre_processor = PreProcess
        self.raw_body_documents = []
        self.body_documents = []
        self.raw_title_documents = []
        self.title_documents = []
        self.ids = None
        self.positional_index_body = None
        self.positional_index_title = None
        self.bigram_index_body = None
        self.bigram_index_title = None

        if self.type == self.TYPE_PERSIAN:
            self.persian_system()
        elif self.type == self.TYPE_ENGLISH:
            self.english_system()
        elif self.type not in self.TYPES:
            raise ValueError('lang should be "english" or "persian"')
        self.initiate_indices()
        self.tf_idf_body = TfIdfSearch(self.ids, self.positional_index_body.index)
        self.tf_idf_title = TfIdfSearch(self.ids, self.positional_index_title.index)

    def initiate_indices(self):
        positional_index_names = self.create_index_names('positional-index', self.type)
        bigram_index_names = self.create_index_names('bigram-index', self.type)

        self.positional_index_body = PositionalIndex(
            name=positional_index_names['body'],
            docs=self.body_documents,
            ids=self.ids
        )
        self.positional_index_title = PositionalIndex(
            name=positional_index_names['title'],
            docs=self.title_documents,
            ids=self.ids
        )

        self.bigram_index_body = BiGramIndex(
            name=bigram_index_names['body'],
            docs=self.body_documents,
            ids=self.ids
        )
        self.bigram_index_title = BiGramIndex(
            name=bigram_index_names['title'],
            docs=self.title_documents,
            ids=self.ids
        )

    @staticmethod
    def create_index_names(index_type, lang_type):
        return {
            doc_type: index_type + '-' + doc_type + '-' + lang_type
            for doc_type in ['title', 'body']
        }

    def persian_system(self):
        tree = ET.parse('../data/Persian.xml')
        namespace = '{http://www.mediawiki.org/xml/export-0.10/}'
        direct_child = namespace + 'page'
        root = tree.getroot()
        body_documents = [
            page.find('{ns}revision/{ns}text'.format(ns=namespace)).text for page in root.findall(direct_child)
        ]
        title_documents = [
            page.find('{ns}title'.format(ns=namespace)).text for page in root.findall(direct_child)
        ]
        self.ids = list(range(len(body_documents)))
        self.pre_processor = PreProcessPersian()
        self.body_documents = self.pre_processor.clean_documents(body_documents)
        self.raw_body_documents = body_documents
        self.title_documents = self.pre_processor.clean_documents(title_documents, with_stop_words=True)
        self.raw_title_documents = title_documents

    def english_system(self):
        doc = pd.read_csv("../data/ted_talks.csv")
        self.pre_processor = PreProcessEnglish()
        self.raw_body_documents = doc['description']
        self.body_documents = self.pre_processor.clean_documents(doc['description'])
        self.ids = list(range(len(self.body_documents)))
        self.title_documents = self.pre_processor.clean_documents(doc['title'], with_stop_words=True)
        self.raw_title_documents = doc['title']

    def retrieve_query_answer(self, query, no_wanted_outcomes, retrieve_type):
        clean_query = self.pre_processor.clean_query(query=query)
        if retrieve_type == 'title':
            handler = self.tf_idf_title
        else:
            handler = self.tf_idf_body
        answers_list = handler.answers(query=clean_query, no_wanted_outcomes=no_wanted_outcomes)
        for i, (score, doc_id) in enumerate(answers_list):
            text = self.retrieve_documents(doc_id, retrieve_type)
            print('{}-answer: {} \n score:{}'.format(i, text, score))

    def retrieve_documents(self, doc_id, retrieve_type):
        if retrieve_type == 'title':
            return self.raw_title_documents[doc_id]
        else:
            return self.raw_body_documents[doc_id]


if __name__ == '__main__':
    ir = IRSystem('english')
    ir.retrieve_query_answer(query='how are you?', no_wanted_outcomes=2, retrieve_type='body')
    now = datetime.datetime.now()
    ir.retrieve_query_answer(query='computer engineering', no_wanted_outcomes=4, retrieve_type='body')
    print(datetime.datetime.now() - now)
