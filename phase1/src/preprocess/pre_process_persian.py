from .pre_process_base import PreProcess
import seaborn as sns
from hazm import Normalizer, word_tokenize, Stemmer
from collections import defaultdict, Counter
from typing import Dict, List


class PreProcessPersian(PreProcess):
    def __init__(self):
        super(PreProcessPersian, self).__init__()
        self.stop_words = None

    def normalization(self, text):
        normalized_text = Normalizer().normalize(text)
        return self.stemming(self.remove_punctuation(self.tokenization(normalized_text)))

    def tokenization(self, text) -> List:
        return word_tokenize(text)

    def find_stopwords(self, documents: List) -> Dict[str, int]:
        normalized_docs = self.normalize(documents)

        words_dict = defaultdict(int)
        for document in normalized_docs:
            for word in document:
                words_dict[word] += 1
        threshold = sum(words_dict.values()) // 200
        counting_doc = Counter()
        for document in normalized_docs:
            counting_doc += Counter(document)

        stop_words_dict = {
            word: count
            for word, count in words_dict.items() if count > threshold
        }

        self.stop_words_dict = stop_words_dict
        self.stop_words = list(stop_words_dict.keys())

        return stop_words_dict

    def stemming(self, tokens):
        stemmer = Stemmer()
        return [stemmer.stem(token) for token in tokens]
