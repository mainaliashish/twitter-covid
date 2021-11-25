"""
-- author: Ashish Mainali
-- email: mainaliashish@outlook.com
-- date: 2021-07-22
"""

import pandas as pd
import fasttext
import os
import numpy as np
import pickle


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


if __name__ == '__main__':
    model_path = os.path.join('data', 'cc.ne.300.bin')

    read_path = os.path.join('fold_dataset', 'set_10', 'test_pad.csv')
    write_path = os.path.join('fold_dataset', 'set_10', 'test_vector_12_300.csv')

    tokens_vectors = dict({
        'Label': [],
        'vectors': []
    })

    # t_c = dict({
    #     'class': [],
    # })
    vector_array = []

    data = pd.read_csv(read_path)
    df = pd.DataFrame(data)
    # print(df)
    # exit(0)
    model = fasttext.load_model(model_path)
    tweets = df['padding_tokens']

    n = 1
    for index, rows in enumerate(tweets):
        tokens = convert_sentences_to_list(rows)
        for _index, token in enumerate(tokens):
            vector = model.get_word_vector(token)
            vector_array.append(vector.tolist())
            # print(token)

        # print(np.shape(vector_array))
        # print(vector_array)
        # exit(0)
        # (12,300) to (300,1)
        mean = np.mean(vector_array, axis=0)
        # mean = mean.reshape(1, 300)
        # print(mean.tolist())
        # exit(0)
        tokens_vectors['Label'].append(df['Label'][index])
        tokens_vectors['vectors'].append(mean.tolist())
        # f_mean = []

        vector_array = []

    my_df = pd.DataFrame(tokens_vectors)
    my_df.to_csv(write_path)
    print(my_df.describe)

print("Successfully generated vector for each tweets..")
