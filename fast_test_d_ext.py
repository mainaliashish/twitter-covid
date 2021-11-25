"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Saturday, July 17, 2021
"""
import os

import fasttext
import pandas as pd
import numpy as np

from tweeter_covid19.utils import mkdir
from tweeter_covid19.utils.pickleutils import read_pickle_data, write_pickle_data

SETS = 10

VECTOR_LEN = 300

if __name__ == '__main__':
    write_path = os.path.join(
        'data', 'fold_train_test_dataset_overall_vectors_for_300dim')

    data_path = os.path.join('fold_dataset')
    model_path = os.path.join('data', 'cc.ne.300.bin')

    # TODO load the fast text model here - optimizer path provide the 1777 dim vector
    # optimizer_path = os.path.join('data', 'training_nodes_vectors')
    #
    # #TODO it is the pointer of optimizer which helps to check if the token is contain in the optimizer or not
    # optimizer_log_path = os.path.join('data', 'training_nodes_vectors')

    # for n_set in range(SETS):
    model = fasttext.load_model(model_path)

    write_joiner_path = os.path.join(write_path, 'set_10')
    read_joiner_path = os.path.join(data_path, "set_10", "train.csv")

    mkdir(write_joiner_path)
    vector_writer_joiner_path = os.path.join(write_joiner_path)
    mkdir(vector_writer_joiner_path)
    data = pd.read_csv(read_joiner_path)

    train_vectors = []
    label_vectors = []
    for index, tweet in enumerate(data['Tweet']):
        if len(tweet.split(' ')) <= 2:
            # here it will ignore less then 2 tweettttts
            continue
        tweet_vectors = np.zeros(
            shape=(len(tweet.split(' ')), VECTOR_LEN), dtype=float)
        checker = 0
        for tweet_token in tweet.split(' '):
            if tweet_token in model:
                try:
                    tweet_vectors[checker] = model.get_word_vector(tweet_token)
                except Exception:
                    pass
                checker += 1
        tweet_vectors = np.average(tweet_vectors, axis=0)
        train_vectors.append(tweet_vectors)
        label_vectors.append(data['Label'][index])
        if index % 1000 == 0:
            print('{}/{} -- Successfully Created! ---- Set : {}'.format(index,
                                                                        len(data['Tweet']), 10
                                                                        ))
    write_pickle_data(os.path.join(
        vector_writer_joiner_path, 'train_x.pkl'), train_vectors)
    write_pickle_data(os.path.join(
        vector_writer_joiner_path, 'train_y.pkl'), label_vectors)
    print("{} - Set CSV file Successfully created")
