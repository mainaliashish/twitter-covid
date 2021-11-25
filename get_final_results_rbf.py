from csv import writer
import numpy as np

import pandas as pd
import os

from sklearn import svm
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    train = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'train_vec_2.csv')
    test = os.path.join('data', 'datazip', 'fold_dataset', 'set_1', 'test_vec_2.csv')
    results_path = os.path.join('data', 'rbf_results_3.csv')

    train_dict = dict({
        'Label': [],
        'train_data': []
    })

    new_train_vec = []

    train = pd.read_csv(train)
    for index, vector in enumerate(train['average_vectors']):
        # print(type(vector))
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

    X_train = np.array(train_dict['train_data'])
    y_train = np.array(train_dict['Label'])

    X_test = np.array(test_dict['test_data'])
    y_test = np.array(test_dict['Label'])

    # print(np.shape(X_train), np.shape(y_train))
    # print(np.shape(X_test), np.shape(y_test))
    # exit(0)

    print("Evaluating results..")

    scaler = StandardScaler()
    # fit scaler on data
    # train_scaler.f(X_train)
    # apply transform
    scaler.fit(X_train)
    # print(standardized)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    svc = svm.SVC(kernel='rbf', class_weight='balanced', C=100, gamma=1)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    print('Accuracy: %f' % accuracy_score(y_test, y_pred))
    print('Recall: %f' % recall_score(y_test, y_pred))
    print('Precision: %f' % precision_score(y_test, y_pred))
    print('F1 Score: %f' % f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    exit(0)

    results = [
        0,
        precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, zero_division=0, average='weighted')]

    append_list_as_row(results_path, results)

    print("Success...")
