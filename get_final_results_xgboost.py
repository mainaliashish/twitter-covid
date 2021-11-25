from csv import writer

import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    train = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'train_3d_vec.csv')
    test = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'test_3d_vec.csv')
    # results_path = os.path.join('data', 'ann_results_17.csv')
    results_path = os.path.join('data', 'xgboost_results_3.csv')

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
        for _index, vect in enumerate(vector):
            vec = float(vect)
            new_test_vec.append(vec)
        test_dict['test_data'].append(new_test_vec)
        test_dict['Label'].append(test['Label'][index])
        new_test_vec = []

    X_train = train_dict['train_data']
    # X_train = X_train
    # print(type([X_train]))
    # exit(0)
    y_train = train_dict['Label']

    X_test = test_dict['test_data']
    y_test = test_dict['Label']

    # # defining parameter range
    # param_grid = {
    #               "max_depth": [5, 6, 7, 10],
    #               "learning_rate": [0.1, 1, 0.01],
    #               "n_estimators": [120, 150, 200]
    #               }
    #
    # grid = GridSearchCV(XGBClassifier(), param_grid, refit=True, verbose=3)
    #
    # # fitting the model for grid
    # # fitting the model for grid search
    # grid.fit(np.array(X_train), np.array(y_train))
    #
    # # print best parameter after tuning
    # print(grid.best_params_)
    #
    # # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    #
    # exit(0)

    # Fit the XgBoost model
    # learning_rate=0.1, max_depth=10, n_estimators=150;, score=0.612 total time=  25.3s
    # learning_rate=1, max_depth=6, n_estimators=200;, score=0.615
    xgb_model = xgb.XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=150)
    xgb_model.fit(np.array(X_train), np.array(y_train))

    y_pred = xgb_model.predict(np.array(X_test))

    print('Accuracy: %f' % accuracy_score(y_test, y_pred))
    print('Recall: %f' % recall_score(y_test, y_pred, average='weighted'))
    print('Precision: %f' % precision_score(y_test, y_pred, zero_division=0, average='weighted'))
    print('F1 Score: %f' % f1_score(y_test, y_pred, zero_division=0, average='weighted'))
    # exit(0)
    results = [
        0,
        precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, zero_division=0, average='weighted')]

    append_list_as_row(results_path, results)

    print("Success...")
