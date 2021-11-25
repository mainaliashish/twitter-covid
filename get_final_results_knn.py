from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from csv import writer
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    results_path = os.path.join('data', 'knn_results_3.csv')

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
    # print(X_train)
    # print(type(X_train))
    # exit(0)
    y_train = train_dict['Label']

    X_test = test_dict['test_data']
    y_test = test_dict['Label']

    # defining parameter range
    # param_grid = {'n_neighbors': [100, 120, 140, 200],
    #               'p': [1, 2, 3],
    #               'leaf_size': [30, 35, 40, 45]
    #               }
    #
    # # C = 1000, gamma = 0.1 kernel = 'linear'
    # # leaf_size=40, n_neighbors=120, p=3;, score=0.577 total time= 1.4min
    # # leaf_size=40, n_neighbors=100, p=3;, score=0.578 total time= 1.1min
    # # leaf_size=30, n_neighbors=100, p=3;, score=0.578 total time= 1.6min
    #
    # grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)
    #
    # # fitting the model for grid
    # # fitting the model for grid search
    # grid.fit(X_train, y_train)
    #
    # # print best parameter after tuning
    # print(grid.best_params_)
    #
    # # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    # exit(0)

    neigh = KNeighborsClassifier(leaf_size=35, n_neighbors=120, p=1)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

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
        f1_score(y_test, y_pred,  zero_division=0, average='weighted')]

    append_list_as_row(results_path, results)
    print("Success...")
