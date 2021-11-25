"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os
from _csv import writer

import numpy as np
# import tensorflow as tf
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.backend import argmax
# from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.ops.confusion_matrix import confusion_matrix

# from tweeter_covid19.classification import Classification
# from tweeter_covid19.utils import standard_normalize, optimize_model
from xgboost import XGBClassifier

from tweeter_covid19.utils.pickleutils import read_pickle_data


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


# def cnn2d_model_for_17features():
#     model = Sequential()
#     model.add(Conv2D(8, kernel_size=(3, 3),
#                      activation='relu',
#                      input_shape=(12, 17, 1)))
#     model.add(Conv2D(8, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(8, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(3, activation='softmax'))
#     return model


N_SETS = 10

if __name__ == '__main__':

    # /home/ashish/word-vector/data
    read_main_path = os.path.join('data', 'fold_train_test_dataset_overall_vectors_for_300dim')
    results_path = os.path.join('data', 'svm_linear_results_300dim.csv')

    for n_set in range(N_SETS):
        print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
        read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))

        train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
        train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
        test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
        test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
        print("---------------- Loading Set {} completed -------------------".format(n_set + 1))

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))
        
        # param_grid = {'C': [1, 100, 100],
        #               'gamma': [0.1, 1, 0.01],
        #               'kernel': ['linear']}
        # # C = 10, gamma = 0.1 kernel = 'linear'
        # grid = GridSearchCV(LinearSVC(), param_grid, refit=True, verbose=2)
        # # fitting the model for grid
        # # fitting the model for grid search
        # grid.fit(train_x, train_y)
        # # print best parameter after tuning
        # print(grid.best_params_)
        # # print how our model looks after hyper-parameter tuning
        # print(grid.best_estimator_)
        # exit(0)

        # model = RandomForestClassifier(min_samples_leaf=3, min_samples_split=6, n_estimators=200)
        # model = XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=150)
        # model = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
        # model = GaussianNB()
        # model = MLPClassifier(hidden_layer_sizes=(20,), learning_rate_init=0.01, max_iter=1000)
        model = KNeighborsClassifier(leaf_size=35, n_neighbors=120, p=1)
        # model = SVC(class_weight='balanced', verbose=1)
        # model.C = 1
        # model.gamma = 0.1
        # model.kernel = 'linear'
        model.fit(train_x, train_y)
        predict = model.predict(test_x)

        print('Accuracy: %f' % accuracy_score(test_y, predict))
        print('Recall: %f' % recall_score(test_y, predict, average="weighted"))
        print('Precision: %f' % precision_score(test_y, predict, average="weighted"))
        print('F1 Score: %f' % f1_score(test_y, predict, average="weighted"))
        print(classification_report(test_y, predict))

        exit(0)

        results = [
            str(n_set + 1),
            accuracy_score(test_y, predict),
            precision_score(test_y, predict, average="weighted"),
            recall_score(test_y, predict, average="weighted"),
            f1_score(test_y, predict, average="weighted")]

        append_list_as_row(results_path, results)
