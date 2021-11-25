"""
 - author : Anish Basnet
 - email : anishbasnetworld@gmail.com
 - date : April 5, 2020
"""
import os

from config import Config
from tweeter_covid19.utils import preprocess_nepali_documents, write_pickle_data, write_text_file_linewise, read_file


class Doc2corpus:
    def __init__(self, stop_word_path=None):
        self.stop_words = None
        self.pairs = None
        self.corpus = dict()
        self.load_stop_words(stop_word_path)
        self.load_pairs()

    def create_corpus(self, data, verbose=False):
        sentences = [data]
        for sentence in sentences:
            # print(sentence)
            if len(sentence) > 0:
                words = preprocess_nepali_documents([sentence], self.stop_words)
                words = list(map(lambda a: a[0], words))
                self.corpus[' '.join(words)] = words
        if verbose:
            print("Successfully created!")

    def fit(self, sent_tokens):
        """
        This function is generally used to get the stemmed corpus of the sentences.
        :param sent_tokens: list
        :return:
        """
        assert type(sent_tokens) == list
        for sentence in sent_tokens:
            tokens = sentence.split(',')
            self.corpus[' '.join(tokens)] = tokens

    def get_corpus(self):
        return self.corpus

    def save_model(self, path):
        write_pickle_data(path, self.corpus)

    def save_corpus(self, path):
        corpus_words = []
        for key in self.corpus:
            corpus_words.append(','.join(self.corpus[key]))
        write_text_file_linewise(file_path=path, data=corpus_words, encoding='utf-8')

    def load_stop_words(self, path=None):
        # TODO : check path type (Exception Handling)
        if path is None:
            path = Config.get_instance()['stop_word_file']
        if os.path.isfile(path):
            self.stop_words = read_file(path)

    def load_pairs(self, path=None):
        if path is None:
            path = Config.get_instance()['rasuwa_dirga_path']
        if os.path.isfile(path):
            self.pairs = read_file(path)
