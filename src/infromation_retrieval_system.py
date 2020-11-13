from src.preprocess import PreProcess, PreProcessEnglish, PreProcessPersian
import pandas as pd
import xml.etree.ElementTree as ET
from src.PositionalIndex import PositionalIndex
from src.BigramIndex import BiGramIndex


class IRSystem:
    TYPE_ENGLISH = 'english'
    TYPE_PERSIAN = 'persian'
    TYPES = [TYPE_ENGLISH, TYPE_PERSIAN]

    def __init__(self, ir_type):
        self.type = ir_type
        self.pre_processor = PreProcess
        self.body_documents = []
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
        titles = [
            page.find('{ns}title'.format(ns=namespace)).text for page in root.findall(direct_child)
        ]
        self.ids = [int(page.find('{ns}id'.format(ns=namespace)).text) for page in root.findall(direct_child)]
        self.pre_processor = PreProcessPersian()
        self.body_documents = self.pre_processor.clean_documents(body_documents)
        self.title_documents = self.pre_processor.clean_documents(titles, with_stop_words=True)

    def english_system(self):
        doc = pd.read_csv("../data/ted_talks.csv")
        self.pre_processor = PreProcessEnglish()
        self.body_documents = self.pre_processor.clean_documents(doc['description'])
        self.ids = list(range(len(self.body_documents)))
        self.title_documents = self.pre_processor.clean_documents(doc['title'], with_stop_words=True)


if __name__ == '__main__':
    ir = IRSystem('english')
    # ir.positional_index_body.show_posting_list('sir')
    # ir.positional_index_title.show_posting_list('you')
    ir.bigram_index_body.show_bigram('pr')
    # print(ir.positional_index_title.index)
    # print(ir.bigram_index_body.index)
    # print(ir.bigram_index_title.index)
