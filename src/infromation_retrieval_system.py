from .preprocess import PreProcessPersian, PreProcessEnglish, PreProcess
import pandas as pd
import xml.etree.ElementTree as ET


class IRSystem:
    def __init__(self, lang):
        self.lang = lang
        self.pre_processor = PreProcess
        self.body_documents = []
        self.title_documents = []

        if lang == 'persian':
            self.persian_system()
        elif lang == 'english':
            self.english_system()
        else:
            raise ValueError('lang should be "english" or "persian"')

    def persian_system(self):
        tree = ET.parse('./data/Persian.xml')
        namespace = '{http://www.mediawiki.org/xml/export-0.10/}'
        root = tree.getroot()
        body_documents = [
            page.find('{ns}revision/{ns}text'.format(ns=namespace)).text for page in root.findall(namespace + 'page')
        ]
        titles = [
            page.find('{ns}title'.format(ns=namespace)).text for page in root.findall(namespace + 'page')
        ]
        self.pre_processor = PreProcessPersian()
        self.body_documents = self.pre_processor.clean_documents(body_documents)
        self.title_documents = self.pre_processor.clean_documents(titles, with_stop_words=True)

    def english_system(self):
        doc = pd.read_csv("data/ted_talks.csv")
        self.pre_processor = PreProcessEnglish()
        self.body_documents = self.pre_processor.clean_documents(doc['description'])
        self.title_documents = self.pre_processor.clean_documents(doc['title'], with_stop_words=True)


if __name__ == '__main__':
    ir = IRSystem('persian')
