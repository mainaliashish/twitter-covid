"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 24, 2020
"""
import numpy as np
from tweeter_covid19.utils import split_word_into_letters, flatten


class Optimizer:
    def __init__(self, labels):
        self.labels = labels
        self.tokens = []
        self.data_structure = dict()

    def process(self, target_label, tokens):
        if not target_label in self.labels:
            return None
        self.tokens.append(tokens)
        self.data_structure[target_label] = dict()
        for token in tokens:
            token_len = len(split_word_into_letters(token))
            if token_len in self.data_structure[target_label]:
                self.data_structure[target_label][token_len].append(token)
            else:
                self.data_structure[target_label][token_len] = [token]

    def get_data_structure(self):
        return self.data_structure

    def get_tokens(self):
        return list(set(flatten(self.tokens)))


class Frequency_generator:
    def __init__(self, model):
        """
        This help to generate the frequency of the tokens.
        :param model: This model should be Optimizer processed model.
        """
        self.unique_tokens = None
        self.model = model
        self.word_with_freq = dict()

    def fit(self):
        self.unique_tokens = self.model.get_tokens()

    def generate_frequency(self, verbose=False):
        labels = self.model.labels
        for index_, token in enumerate(self.unique_tokens):
            token_len = len(split_word_into_letters(token))
            frequency = np.zeros(len(labels), dtype=int)
            for index, label in enumerate(labels):
                tokens_with_count = self.model.data_structure[label]
                if token_len in tokens_with_count:
                    for value in tokens_with_count[token_len]:
                        if value == token:
                            frequency[index] += 1
            self.word_with_freq[token] = frequency
            if verbose and index_ % 1000 == 0:
                print('Token - {} . Successfully Executed - Current : '
                      '{}/{} .'.format(token, index_, len(self.unique_tokens)))
        return self.word_with_freq
