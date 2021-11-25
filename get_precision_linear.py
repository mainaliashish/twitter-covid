from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import os
import numpy as np


if __name__ == '__main__':
    train = os.path.join('fold_dataset', 'set_2', 'train_v_2_vector.csv')
    test = os.path.join('fold_dataset', 'set_2', 'test_v_2_vector.csv')

    train_data = pd.read_csv(train, index_col=0)
    print(train_data.head())
    print(train_data.groupby('Label').size())
    X = train_data.drop(['Label'], axis=1)
    print(X.head())
    exit(0)
    # train_df = pd.DataFrame(train_data)
    # print(train_data.head())
    # print(train_data.describe())
    X = train_data.drop(['Label', 'tweets', 'tokens'], axis=1)
    y = train_data['vectors_average']
    print(X.shape)
    print(y.shape)
    # print(X)
    exit(0)

    test_data = pd.read_csv(test)
    test_df = pd.DataFrame(test_data)

    predicted = []
    y_test = []

    precision, recall, fscore, support = score(y_test, predicted)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))