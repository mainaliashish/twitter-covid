import pickle
import os

import numpy as np
import pandas as pd


def convert_tokens_to_list(string):
    li = list(string.split(","))
    return li


if __name__ == '__main__':
    logger_path = os.path.join('data', 'datazip', 'logger.pkl')
    training_path = os.path.join('data', 'datazip', 'direct_training_vectors_final.pkl')
    train_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'test_pad.csv')
    write_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'test_vec_2.csv')

    all_vectors = open(training_path, 'rb')
    all_vectors = pickle.load(all_vectors)
    logger = open(logger_path, 'rb')
    logger = pickle.load(logger)

    # if 'कौभीड़' in logger and 'कौभीड़' in all_vectors:
    #     print(all_vectors['कौभीड़'])
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
        if len(n_tokens) > 2 and n_tokens[0] in all_vectors and n_tokens[1] in all_vectors:
            for _index, token in enumerate(n_tokens):
                if token in logger and token in all_vectors:
                    print(token)
                    # print(all_vectors[token])
                    new_vec.append(all_vectors[token])
                    # print(all_vectors[token])
                    # exit(0)
                else:
                    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    new_vec.append(np.array(a))
                    continue
        else:
            continue
        x = np.array(new_vec)
        # exit(0)
        # print(type(x))
        mean = np.mean(x, axis=0)
        # print(mean)
        # exit(0)
        new_dict['Label'].append(train_data['Label'][index])
        new_dict['average_vectors'].append(mean.tolist())
        new_vec = []
    exit(0)
    my_df = pd.DataFrame(new_dict)
    my_df.to_csv(write_path)
    print(my_df.describe)
    print("Success..")
