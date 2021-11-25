"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from tweeter_covid19.classification import Classification
from tweeter_covid19.utils import standard_normalize, optimize_model
from tweeter_covid19.utils.pickleutils import read_pickle_data


def cnn2d_model_for_17features():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(12, 17, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


N_SETS = 10

if __name__ == '__main__':

    read_main_path = os.path.join('data', 'fold_train_test_collector_with_normalization')

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

        train_x = np.average(train_x, axis=1)
        test_x = np.average(test_x, axis=1)

        print(np.shape(train_x), np.shape(train_y))
        print(np.shape(test_x), np.shape(test_y))

        model = SVC()
        model.C = 50
        model.gamma = 1e-04
        model.kernel = 'rbf'
        model.fit(train_x, train_y)
        predict = model.predict(test_x)

        print(classification_report(test_y, predict))
        exit(0)
