from .pre_process_base import PreProcess

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict, Counter
from typing import List, Dict


class PreProcessEnglish(PreProcess):
    def __init__(self):
        super(PreProcessEnglish, self).__init__()
        self.stop_words = []

    def normalization(self, text):
        return self.stemming(self.remove_punctuation(self.tokenization(text)))

    def tokenization(self, text) -> List:
        tokens = word_tokenize(text)
        return tokens

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
            for word, count in words_dict.items() if word in stopwords.words("english")[1:100] and count > threshold
        }
        self.stop_words_dict = stop_words_dict
        self.stop_words = list(stop_words_dict.keys())

        return stop_words_dict

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def stemming(self, tokens):
        stemmer = SnowballStemmer(language='english')
        return [stemmer.stem(token) for token in tokens]
