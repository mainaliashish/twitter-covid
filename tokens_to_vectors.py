"""
-- author: Ashish Mainali
-- email: mainaliashish@outlook.com
-- date: 2021-07-17
"""

import pandas as pd
import fasttext
import os
import pickle


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


if __name__ == '__main__':
    model_path = os.path.join('data', 'cc.ne.300.bin')
    read_path = os.path.join('data', 'testdata.csv')
    pickle_path = os.path.join('data', 'words_vectors.pickle')

    tokens_vectors = dict({
        'token': [],
        'vectors': []
    })

    data = pd.read_csv(read_path)
    df = pd.DataFrame(data)
    model = fasttext.load_model(model_path)
    tweets = df['Tokanize_tweet']
    for index, tokens in enumerate(tweets):
        tokens = convert_sentences_to_list(tokens)
        for token in tokens:
            vector = model.get_word_vector(token)
            tokens_vectors['token'].append(token)
            tokens_vectors['vectors'].append(vector)

        pickle.dump(tokens_vectors, open(pickle_path, "wb"))

tokens = pickle.load(open(pickle_path, "rb"))
print(tokens)
