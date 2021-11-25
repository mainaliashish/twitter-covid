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
    read_path = os.path.join('words_frequency.csv')
    logger_path = os.path.join('data', '300D_vectors', 'logger.pkl')
    final_path = os.path.join('data', '300D_vectors', 'all_300d_vectors.pkl')

    tokens_vectors = dict({
        'key': [],
        'vectors': []
    })
    #
    # pickle_map = dict({
    #     'key': []
    # })

    vector_array = []
    pickle_map = []

    data = pd.read_csv(read_path)
    df = pd.DataFrame(data)

    model = fasttext.load_model(model_path)

    words = df['words']
    for index, word in enumerate(words):
        # print(len(word))
        # exit(0)
        if len(word) >= 1 and word in model:
            vector = model.get_word_vector(word)
            # print(vector)
            # exit(0)
            # else:
            # vector = np.zeros(300)
            vector_dict = dict({
                df['words'][index]: vector.tolist()
            })
            # tokens_vectors['key'].append(df['words'][index])
            # tokens_vectors['vectors'].append(vector.tolist())

            # print(vector_dict)
            # exit(0)
            #
            pickle_map.append(df['words'][index])
            # print(pickle_map)
            # exit(0)

            # print(tokens_vectors)
            # exit(0)
            with open(final_path, 'ab+') as fp:
                pickle.dump(vector_dict, fp)
                fp.close()
            vector_dict.clear()

    f = open(logger_path, 'wb')
    pickle.dump(pickle_map, f)

    # tweets = df['padding_tokens']
    # n = 1
    # for index, rows in enumerate(tweets):
    #     tokens = convert_sentences_to_list(rows)
    #     for _index, token in enumerate(tokens):
    #         vector = model.get_word_vector(token)
    #         vector_array.append(vector.tolist())
    #
    #     print(vector_array)
    #     exit(0)
    #     tokens_vectors['Label'].append(df['Label'][index])
    #     tokens_vectors['key'].append(df['old_tokens'][index])
    #     tokens_vectors['vectors'].append(vector_array)
    #     # print(tokens_vectors['vectors'])
    #     # exit(0)
    #     pickle_path = os.path.join('data', '300_vectors', 'vectors_300_'+str(n))
    #     f = open(pickle_path, 'wb')
    #     pickle.dump(tokens_vectors, f)
    #
    #     pickle_map['key'].append(df['old_tokens'][index])
    #     pickle_map['path'].append(pickle_path)
    #     logger_path = os.path.join('data', '300_vectors', 'logger.pkl')
    #     f = open(logger_path, 'ab+')
    #     pickle.dump(pickle_map, f)
    #
    #     n += 1
    #     vector_array = []
    #     tokens_vectors = dict({
    #         'Label': [],
    #         'key': [],
    #         'vectors': []
    #     })
    #     pickle_map = dict({
    #         'key': [],
    #         'path': []
    #     })
    # exit(0)

    # my_df = pd.DataFrame(tokens_vectors)
    # my_df.to_csv(write_path)
    # print(my_df.describe)
    # print(pickle_map)
print("Successfully generated vector for each tweets..")
