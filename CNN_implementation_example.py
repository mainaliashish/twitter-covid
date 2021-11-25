"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : Sunday, July 19, 2021
"""
import os

import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
import keras.backend as K
from matplotlib import pyplot as plt

# def cnn_model():
#     sequence_input = Input(shape=(300, 1))
#     x = Conv1D(128, 5, activation='relu')(sequence_input)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)
#     x = Dropout(0.7)(x)
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     preds = Dense(3, activation='softmax')(x)
#     return Model(sequence_input, preds)

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


def cnn_model_for_17dim():
    sequence_input = Input(shape=(17, 1))
    x = Conv1D(8, 2, activation='relu')(sequence_input)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)  # global max pooling
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(8, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)
    return Model(sequence_input, preds)


# def create_model(optimizer='rmsprop'):
#     model = Sequential()
#     model.add(Dense(512, input_shape=(300, 1)))
#     model.add(Conv1D(8, 2, activation='relu'))
#     model.add(MaxPooling1D(2))
#     model.add(Activation('relu'))  # An "activation" is just a non-linear function applied to the output
#     model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
#     model.add(Dense(512,))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(10,))
#     model.add(Activation('softmax'))  # This special "softmax" a
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model


def convert_average_to_list(string):
    li = list(string.split(","))
    return li


def recall_m(y_train, y_test):
    # y_train = K.ones_like(y_train)
    true_positives = K.sum(K.round(K.clip(y_train * y_test, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_train, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision_m(y_train, y_test):
    # y_train = K.ones_like(y_train)
    true_positives = K.sum(K.round(K.clip(y_train * y_test, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_train, y_test):
    precision = precision_m(y_train, y_test)
    recall = recall_m(y_train, y_test)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == '__main__':
    model = cnn2d_model()
    # print(model.summary())
    # exit(0)
    # model_2 = cnn_model_for_17dim()
    # print(model_2.summary())

    train = os.path.join('fold_dataset', 'set_1', 'train_vector_12_300.csv')
    test = os.path.join('fold_dataset', 'set_1', 'test_vector_12_300.csv')

    # train_dict = dict({
    #     'Label': [],
    #     'train_data': []
    # })
    #
    # new_train_vec = []
    #
    train = pd.read_csv(train)
    # for index, vector in enumerate(train['vectors']):
    #     # vector = vector.strip('[]')
    #     # print(np.array(vector))
    #     # print(type(np.array(vector)))
    #     # exit(0)
    #     vector = np.array(vector)
    #     for _index, vect in enumerate(vector):
    #         print(vect)
    #         print(type(vect))
    #         exit(0)
    #         vect = vect.strip('[]')
    #         print(vect)
    #         vec = float(vect)
    #         print(vec)
    #         new_train_vec.append(vec)
    #     train_dict['train_data'].append(new_train_vec)
    #     train_dict['Label'].append(train['Label'][index])
    #     new_train_vec = []
    # exit(0)
    # train_data = train_dict['train_data']
    #
    test_dict = dict({
        'Label': [],
        'test_data': []
    })
    new_test_vec = []
    test = pd.read_csv(test)
    for index, vector in enumerate(test['vectors']):
        # vector = vector.strip('[]')
        # print(vector)
        # exit(0)
        # vector = convert_average_to_list(vector)
        print(type(vector))
        exit(0)
        for _index, vect in enumerate(vector):
            # vec = float(vect)
            new_test_vec.append(vect)
        test_dict['test_data'].append(new_test_vec)
        test_dict['Label'].append(test['Label'][index])
        new_test_vec = []

    X_train = train['vectors']
    # print(X_train)
    # print(np.shape(X_train))
    # exit(0)
    y_train = np.array(train['Label'])

    X_test = test_dict['test_data']
    # for index, item in enumerate(X_test):
    #     print(item)
    #     print(np.shape(item))
    #     exit(0)
    # exit(0)
    print(X_test)
    print(np.shape(X_test))
    exit(0)
    y_test = np.array(test['Label'])

    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    # print(y_train)
    # exit(0)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(np.shape(X_train), np.shape(y_train))
    print(np.shape(X_test), np.shape(y_test))
    exit(0)
    # optimizers = ['SGD', 'RMSprop', 'Adam']
    # learning_rate = 0.0001
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    # , precision_m, recall_m, f1_score
    # happy learning!
    # K.set_value(model.optimizer.learning_rate, 0.001)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=128)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.show()

    # history = model.fit(X_train, y_train,validation_split=0.1, epochs=50, batch_size=128)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
