from csv import writer

import numpy as np
import pandas as pd
import os

from sklearn import svm, preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    train = os.path.join('fold_dataset', 'set_1', 'train_vector_12_300.csv')
    test = os.path.join('fold_dataset', 'set_1', 'test_vector_12_300.csv')
    # test = os.path.join('fold_dataset', 'set_10', 'test_vector.csv')
    results_path = os.path.join('data', 'svm_linear_results_3.csv')

    train_dict = dict({
        'Label': [],
        'train_data': []
    })

    new_train_vec = []

    train = pd.read_csv(train)
    for index, vector in enumerate(train['vectors']):
        vector = vector.strip('[]')
        vector = convert_average_to_list(vector)
        for _index, vect in enumerate(vector):
            # print(vect)
            # exit(0)
            vec = float(vect)
            new_train_vec.append(vec)
        train_dict['train_data'].append(new_train_vec)
        train_dict['Label'].append(train['Label'][index])
        new_train_vec = []

    # print(train_dict['train_data'])
    # print(type(train_dict['train_data']))
    # exit(0)
    test_dict = dict({
        'Label': [],
        'test_data': []
    })
    new_test_vec = []
    test = pd.read_csv(test)
    # print(test.head())
    # exit(0)
    for index, vector in enumerate(test['vectors']):
        # print(vector)
        # exit(0)
        vector = vector.strip('[]')
        vector = convert_average_to_list(vector)
        for _index, vect in enumerate(vector):
            # print(type(vect))
            # exit(0)
            vec = float(vect)
            # print(vec)
            # exit(0)
            new_test_vec.append(vec)
        test_dict['test_data'].append(new_test_vec)
        test_dict['Label'].append(test['Label'][index])
        new_test_vec = []

    # train_data = pd.DataFrame(train_dict)
    # print(train_data.groupby('Label').size())
    # exit(0)

    X_train = np.array(train_dict['train_data'])
    # print(np.shape(X_train))
    # exit(0)
    y_train = np.array(train_dict['Label'])
    y_train = y_train.reshape(len(y_train))
    # print(y_train)
    # print(X_train)
    # print(type(y_train))
    # exit(0)
    # print(test_dict['test_data'])
    # exit(0)
    X_test = np.array(test_dict['test_data'])
    y_test = np.array(test_dict['Label'])
    y_test = y_test.reshape(len(y_test))
    # print(X_test)
    # print(type(X_test))
    # exit(0)

    # # #############################################################################
    # # Train classifiers
    # #
    # # # # defining parameter range
    # param_grid = {'C': [1, 100, 1000],
    #               'gamma': [0.1, 1, 0.01],
    #               'kernel': ['linear']}
    # # C = 10, gamma = 0.1 kernel = 'linear'
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # # fitting the model for grid
    # # fitting the model for grid search
    # grid.fit(X_train, y_train)
    # # print best parameter after tuning
    # print(grid.best_params_)
    # # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    # # exit(0)
    # grid_predictions = grid.predict(X_test)
    # # print classification report
    # # print(classification_report(y_test, grid_predictions))
    # exit(0)
    print("Evaluating results..")

    # scaler = StandardScaler()
    # # fit scaler on data
    # # train_scaler.f(X_train)
    # # apply transform
    # scaler.fit(X_train)
    # # print(standardized)
    # X_train = scaler.transform(X_train)
    #

    # X_test = scaler.transform(X_test)

    # from sklearn.preprocessing import normalize
    # normalize(X)

    scaler = StandardScaler()
    # fit scaler on data
    # train_scaler.f(X_train)
    # apply transform
    scaler.fit(X_train)
    # print(standardized)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    # x_test_scaler = StandardScaler()
    # # fit scaler on data
    # x_test_scaler.fit(X_test)
    # # apply transform
    # X_test = x_test_scaler.transform(X_test)
    print("Transformed inputs to standard scaler")

    # le = preprocessing.LabelEncoder()
    # le.fit(y_train)
    #
    # y_train = le.transform(y_train)
    # y_test = le.transform(y_test)
    # print(y_train)
    # exit(0)

    # test_scaler = StandardScaler()
    # # fit scaler on data
    # test_scaler.fit(y_train)
    # # apply transform
    # y_train = test_scaler.transform(y_train)
    #
    # y_test_scaler = StandardScaler()
    # # fit scaler on data
    # y_test_scaler.fit(y_test)
    # # apply transform
    # y_test = y_test_scaler.transform(y_test)
    print("Fitting and transforming train and test vectors..")
    svc = svm.SVC(C=10, kernel='linear', gamma=0.1, random_state=1, verbose=2)
    svc.fit(X_train, y_train)
    print("Evaluating Results..")

    y_pred = svc.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    # exit(0)

    print('Accuracy: %f' % accuracy_score(y_test, y_pred))
    print('Recall: %f' % recall_score(y_test, y_pred, average='weighted'))
    print('Precision: %f' % precision_score(y_test, y_pred, zero_division=0, average='weighted'))
    print('F1 Score: %f' % f1_score(y_test, y_pred, zero_division=0, average='weighted'))
    # exit(0)
    # print(classification_report(y_test, y_pred))
    # print()
    exit(0)

    results = [
        0,
        precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, zero_division=0, average='weighted')]

    append_list_as_row(results_path, results)

    print("Success...")



