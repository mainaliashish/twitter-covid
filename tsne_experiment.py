"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""

import os
from _csv import writer

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    train = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'train_3d_vec_full.csv')
    test = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'test_3d_vec_full.csv')
    results_path = os.path.join('data', 'naive_results_3.csv')
    write_path = os.path.join('data', 'embedding_eval_1.png')

    train_dict = dict({
        'Label': [],
        'train_data': []
    })

    new_train_vec = []

    train = pd.read_csv(train)
    for index, vector in enumerate(train['average_vectors']):
        vector = vector.strip('[]')
        vector = convert_average_to_list(vector)
        for _index, vect in enumerate(vector):
            vec = float(vect)
            new_train_vec.append(vec)
        train_dict['train_data'].append(new_train_vec)
        train_dict['Label'].append(train['Label'][index])
        new_train_vec = []

    train_data = train_dict['train_data']

    test_dict = dict({
        'Label': [],
        'test_data': []
    })
    new_test_vec = []
    test = pd.read_csv(test)
    for index, vector in enumerate(test['average_vectors']):
        vector = vector.strip('[]')
        vector = convert_average_to_list(vector)
        for _index, t_vect in enumerate(vector):
            vec = float(t_vect)
            new_test_vec.append(vec)
        test_dict['test_data'].append(new_test_vec)
        test_dict['Label'].append(test['Label'][index])
        new_test_vec = []

    X_train = np.array(train_dict['train_data'])
    # X_train = X_train
    # print(type([X_train]))
    # exit(0)
    y_train = np.array(train_dict['Label'])

    X_test = np.array(test_dict['test_data'])
    y_test = np.array(test_dict['Label'])

    palette_colors = len(set(y_train))

    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(X_train)

    tsne_df = pd.DataFrame({'X': x_2d[:, 0],
                            'Y': x_2d[:, 1],
                            'Classes': y_train})
    tsne_df.head()
    fig = sns.scatterplot(x=x_2d[:, 0], y=x_2d[:, 1],
                          hue=y_train,
                          palette=sns.color_palette("Set1", n_colors=palette_colors, desat=.5),
                          legend=False)
    fig.figure.savefig(write_path)
