import pickle
import os

import numpy as np
import pandas as pd


def convert_tokens_to_list(string):
    li = list(string.split(","))
    return li


if __name__ == '__main__':
    # /home/ashish/word-vector/training_nodes_vectors/set_1
    logger_path = os.path.join('data', '300D_vectors', 'logger.pkl')
    training_path = os.path.join('data', '300D_vectors', 'all_300d_vectors.pkl')
    train_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_3', 'test_pad.csv')
    write_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_3', 'test_3d_vec.csv')

    all_vectors = open(training_path, 'rb')
    all_vectors = pickle.load(all_vectors)
    logger = open(logger_path, 'rb')
    logger = pickle.load(logger)
    # test = pd.read_pickle(logger)
    print(all_vectors['कोभिड'])
    exit(0)
    # exit(0)
    # for index, vec in enumerate(all_vectors):
    #     print(vec)
    #     exit(0)
    # exit(0)

    # if 'प्रदेश' in logger and 'प्रदेश' in all_vectors:
    #     print(all_vectors['प्रदेश'])
    #     print('Found')
    # else:
    #     print('Not found')
    #
    # exit(0)

    new_dict = ({
        'Label': [],
        # 'Tokenize_tweets': [],
        'average_vectors': []
    })

    train_data = pd.read_csv(train_path)
    new_vec = []
    for index, tokens in enumerate(train_data['padding_tokens']):
        n_tokens = convert_tokens_to_list(tokens)
        # print(len(n_tokens))
        # print(n_tokens[0])
        # exit(0)
        if len(n_tokens) > 2 and n_tokens[0] in all_vectors and n_tokens[1] in all_vectors:
            for _index, token in enumerate(n_tokens):
                if token in logger and token in all_vectors:
                    print(token)
                    # print(all_vectors[token])
                    new_vec.append(all_vectors[token])
                    # print(np.shape(new_vec))
                else:
                    a = np.zeros(300)
                    new_vec.append(np.array(a))
                    continue
        else:
            continue

        # print(new_vec)
        # exit(0)
        mean = np.mean(new_vec, axis=0)
        # print(mean)
        # exit(0)

        new_dict['Label'].append(train_data['Label'][index])
        new_dict['average_vectors'].append(mean.tolist())
        new_vec = []

    my_df = pd.DataFrame(new_dict)
    my_df.to_csv(write_path)

    print(my_df.describe)
    print("Success")
