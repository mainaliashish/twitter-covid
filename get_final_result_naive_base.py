import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from csv import writer

import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    train = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'train_3d_vec.csv')
    test = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'test_3d_vec.csv')
    results_path = os.path.join('data', 'naive_results_3.csv')

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

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # exit(0)

    # nb_classifier = GaussianNB()
    #
    # params_NB = {'var_smoothing': [i for i in range(1, 50)]}
    # gs_NB = GridSearchCV(estimator=nb_classifier,
    #                      param_grid=params_NB,
    #                      # cv=cv_method,  # use any cross validation technique
    #                      verbose=3,
    #                      scoring='accuracy')
    # gs_NB.fit(X_train, y_train)
    #
    # print(gs_NB.best_params_)
    # exit(0)
    gnb = GaussianNB(priors=None, var_smoothing=1)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    print('Precision: %f' % precision_score(y_test, y_pred, zero_division=0, average='weighted'))
    print('Accuracy: %f' % accuracy_score(y_test, y_pred))
    print('Recall: %f' % recall_score(y_test, y_pred, average='weighted'))
    print('F1 Score: %f' % f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))
    # print()
    exit(0)
    # exit(0)

    results = [
        0,
        precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred,  zero_division=0, average='weighted')]

    append_list_as_row(results_path, results)

    print("Success...")
