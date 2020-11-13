from src.preprocess import PreProcess, PreProcessEnglish, PreProcessPersian
import pandas as pd
import xml.etree.ElementTree as ET


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

        if ir_type == self.TYPE_PERSIAN:
            self.persian_system()
        elif ir_type == self.TYPE_ENGLISH:
            self.english_system()
        elif ir_type not in self.TYPES:
            raise ValueError('lang should be "english" or "persian"')

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
    ir = IRSystem('persian')
