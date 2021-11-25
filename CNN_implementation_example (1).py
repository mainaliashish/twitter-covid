"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical

from tweeter_covid19.utils.pickleutils import read_pickle_data

from sklearn import preprocessing


def cnn2d_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(12, 300, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


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

if __name__ == '__main__':
    model = cnn2d_model_for_17features()
    print(model.summary())

    # read_main_path = os.path.join('data', 'fold_train_test_collector')
    #
    # for n_set in range(N_SETS):
    #     print("---------------- Loading Set {} data sets -------------------".format(n_set + 1))
    #     read_joiner_path = os.path.join(read_main_path, 'set_' + str(n_set + 1))
    #
    #     train_x = read_pickle_data(os.path.join(read_joiner_path, 'train_x.pkl'))
    #     train_y = read_pickle_data(os.path.join(read_joiner_path, 'train_y.pkl'))
    #     test_x = read_pickle_data(os.path.join(read_joiner_path, 'test_x.pkl'))
    #     test_y = read_pickle_data(os.path.join(read_joiner_path, 'test_y.pkl'))
    #     print("---------------- Loading Set {} completed -------------------".format(n_set + 1))
    #
    #     print(np.shape(train_x), np.shape(train_y))
    #     print(np.shape(test_x), np.shape(test_y))
    #
    #     le = preprocessing.LabelEncoder()
    #     le.fit(train_y)
    #
    #
    #
    #     train_y = le.transform(train_y)
    #     test_y = le.transform(test_y)
    #
    #     train_y = to_categorical(train_y)
    #     test_y = to_categorical(test_y)
    #
    #     train_x = np.array(train_x).reshape((np.shape(train_x)[0], np.shape(train_x)[1], np.shape(train_x)[2], 1))
    #     test_x = np.array(test_x).reshape((np.shape(test_x)[0], np.shape(test_x)[1], np.shape(test_x)[2], 1))
    #
    #     print(np.shape(train_x), np.shape(train_y))
    #     print(np.shape(test_x), np.shape(test_y))

    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['acc'])
    #
    # # happy learning!
    # model.fit(train_x, train_y, validation_data=(test_x, test_y),
    #           epochs=2, batch_size=128)
