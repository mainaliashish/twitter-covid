"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 15, 2020
"""
import os
import re

from tweeter_covid19.utils import flatten, write_pickle_data, read_pickle_data, write_text_file_linewise


class GraphGenerator:
    def __init__(self, corpus=None):
        self.corpus = corpus
        self.tokens = None
        self.graph_model = None
        self.graph_path = None

    def fit(self, graph_path=None, write_path=None):
        if not os.path.isfile(write_path):
            self.tokens = list(set(flatten(list(self.corpus.values()))))
            words = []
            for token in self.tokens:
                if re.search(pattern='।|’|,|‘', string=token):
                    match_regular_expression = re.findall(pattern='।|’|,|‘|', string=token)
                    for matched_string in match_regular_expression:
                        token = re.sub(matched_string, '', token).strip()
                if len(token) > 1 and not re.search(pattern='[०१२३४५६७८९]', string=token):
                    words.append(token)
            self.tokens = words
            write_pickle_data(write_path, words)
        else:
            self.tokens = read_pickle_data(write_path)
        self.graph_path = graph_path

    def search(self, token, current, total):
        if re.search(r'\(|\)|\.|\?|\*|\\|/|//|\\\\|[०१२३४५६७८९]|[0-9]|"|\[', token):
            print('Token Skipped!!!!!! - {} -- Current token - {}/{} .'.format(token, current, total))
            return []
        link_list = [token]
        for key in self.corpus:
            try:
                pattern = re.compile(token)
                if pattern.findall(key):
                    link_list.append(self.corpus[key])
            except Exception:
                print("Token Error -  {} -- Current token - {}/{} .".format(token, current, total))
        print('Token - {} -- Current token - {}/{} .'.format(token, current, total))
        return flatten(link_list)

    def connect_edges(self, log_path=None):
        log = []
        if os.path.isfile(log_path):
            log = read_pickle_data(log_path)
        for index, token in enumerate(self.tokens):
            if token in log:
                print("Token already Processed!")
            else:
                edges = self.search(token, index, len(self.tokens))
                if edges:
                    write_text_file_linewise(os.path.join(self.graph_path, str(index) + ".txt"), edges, encoding='utf-8')
                    log.append(token)
                    write_pickle_data(log_path, log)
