from src.preprocess import PreProcessEnglish, PreProcessPersian
import pandas as pd
import xml.etree.ElementTree as ET
from src.PositionalIndex import PositionalIndex
from src.BigramIndex import BiGramIndex
from src.tf_idf import TfIdfSearch
from src.QueryCorrection import correct_query
from src.ProximitySearch import proximity_search


class IRSystem:
    TYPE_ENGLISH = 'english'
    TYPE_PERSIAN = 'persian'
    TYPES = [TYPE_ENGLISH, TYPE_PERSIAN]

    def __init__(self, ir_type):
        self.type = ir_type
        self.pre_processor = None
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

    def retrieve_tfidf_answer(self, query, no_wanted_outcomes, retrieve_type):
        print('your query: {}'.format(query))
        clean_query = self.pre_processor.clean_query(query=query)
        if retrieve_type == 'title':
            handler = self.tf_idf_title
        else:
            handler = self.tf_idf_body
        answers_list = handler.answers(query=clean_query, no_wanted_outcomes=no_wanted_outcomes)
        print('tf-idf search:')
        for i, (score, doc_id) in enumerate(answers_list):
            text = self.retrieve_documents(doc_id, retrieve_type)
            print('{}-answer: {} \n score:{}'.format(i, text, score))

    def retrieve_documents(self, doc_id, retrieve_type):
        if retrieve_type == 'title':
            return self.raw_title_documents[doc_id]
        else:
            return self.raw_body_documents[doc_id]

    def retrieve_proximity_answer(self, query, window, retrieve_type):
        print('your query: {}'.format(query))
        clean_query = self.pre_processor.tokenization(query)
        if retrieve_type == 'title':
            handler = self.positional_index_title
        else:
            handler = self.positional_index_body
        answers_list = proximity_search(self.ids, clean_query, handler.index, window)
        print('proximity search result:')
        for i, (score, doc_id) in enumerate(answers_list):
            text = self.retrieve_documents(doc_id, retrieve_type)
            print('{}-answer: {} \n score:{}'.format(i, text, score))

    def corrected_query(self, query):
        q = correct_query(query, self.positional_index_body.index)
        print("corrected query:{}".format(q))
        return q

    def plot_stop_words(self):
        self.pre_processor.plot_stop_words()


if __name__ == '__main__':
    ir = IRSystem('english')

    query = 'how ar yoo my broter'
    # if user wants to corrects the query by IR system
    ir.corrected_query(query)

    # if user want to see deleted stopwords:
    ir.plot_stop_words()

    # for tf_idf_search
    ir.retrieve_tfidf_answer(query='how are you?', no_wanted_outcomes=2, retrieve_type='body')
    ir.retrieve_tfidf_answer(query='computer engineering', no_wanted_outcomes=4, retrieve_type='body')

    # for proximity search
    ir.retrieve_proximity_answer(query='what is makes?', window=5, retrieve_type='body')
