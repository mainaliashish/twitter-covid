"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""
import ast
import os
import numpy as np
from pandas_ml import ConfusionMatrix
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from news_classification import read_pickle_data
from news_classification.utils import get_all_directory
from news_classification.classification import Classification
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def get_data_split(data):
    x = []
    y = []
    for key, value in data.items():
        target = list(value.keys())[0]
        x.append(value[target])
        y.append(target)
    return x, y


def get_data_from_csv(path, header):
    df = pd.read_csv(path)
    return list(df[header])

# cm = confusion_matrix(classes, predict)
# print(cm)
# print(classification_report(classes, predict))
# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.xlabel('Predicted labels \nAccuracy: {:0.2f}'.format(acc * 100))
# plt.ylabel("True labels")
# plt.xticks(classes, [])
# plt.yticks(classes, [])
# plt.title('Confusion matrix ')
# plt.colorbar()
# plt.show()


if __name__ == '__main__':
    title_ = "nepali_news_dataset_20_categories_large"
    labels_path = os.path.join('data', 'datasets', 'scrap_dataset',
                               'sets', 'set_1', 'test')

    train_path = os.path.join('data', 'vectors', 'distance_based_histogram', 'nepali_linguistic',
                              'sets', 'set_1',
                              'avg_distance_based_train.pkl')
    test_path = os.path.join('data', 'vectors', 'distance_based_histogram', 'nepali_linguistic',
                             'sets', 'set_1',
                             'avg_distance_based_test.pkl')
    # write_path = os.path.join('data', 'vectors', 'distance_based_histogram', 'nepali_linguistic',
    #                           'confusion_matrix.svg')

    train_data = read_pickle_data(train_path)
    test_data = read_pickle_data(test_path)

    train_x, train_y = get_data_split(train_data)
    test_x, test_y = get_data_split(test_data)
    # labels_path = os.path.join('data', 'datasets', 'fusion_news', 'sets')
    #
    # train_path = os.path.join('data', 'vectors', 'distance_based_histogram', '24NepaliNews',
    #                           'avg_distance_based_train.csv')
    # test_path = os.path.join('data', 'vectors', 'distance_based_histogram', '24NepaliNews',
    #                          'avg_distance_based_test.csv')

    #
    # train_x = [ast.literal_eval(x) for x in get_data_from_csv(train_path, 'vector')]
    # test_x = [ast.literal_eval(x) for x in get_data_from_csv(test_path, 'vector')]
    # train_y = get_data_from_csv(train_path, 'label')
    # test_y = get_data_from_csv(test_path, 'label')

    print(np.shape(train_x), np.shape(train_y))
    print(np.shape(test_x), np.shape(test_y))

    model = Classification()
    model.fit_parameter(c=81, kernel='rbf', gamma=1e-04)

    prediction, confusion_matrix_ = model.get_confusion_matrix(train_x, train_y, test_x, test_y, normalize=True)
    targets = get_all_directory(labels_path)
    # Conf_mat = ConfusionMatrix(test_y, prediction)
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 150)
    # print(Conf_mat.print_stats())
    # print(len(targets))
    # print(classification_report(test_y, prediction))
    # exit(0)

    confusion_matrix_df = pd.DataFrame(confusion_matrix_, index=targets, columns=targets)
    fig = plt.figure(figsize=(3.50, 2.50), dpi=100)
    sn.set(font_scale=0.5)
    # cmap = sn.color_palette("Set1", n_colors=len(targets)*len(targets), desat=.5)
    sn.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 8}, cmap="OrRd")  # font size
    plt.title(title_)
    # plt.savefig(write_path, format='svg', dpi=9400)
    plt.show()
