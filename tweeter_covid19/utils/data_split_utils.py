"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : March 15, 2020
"""
import os

import numpy as np

from tweeter_covid19.utils import get_all_files


def split_document_based_data_set(split_documents=None, ratio=None, verbose=False):
    total_document = len(split_documents)
    total_train_document = int(ratio * total_document)
    split_train = np.random.choice(split_documents, total_train_document, replace=False)
    split_test = np.array(list(set(split_documents).difference(set(split_train))))
    if verbose:
        print("{} - {} -> Total : {}\nTrain Files - {}\n Test Files -{}"
              .format(len(split_train), len(split_test), total_document, split_train, split_test))
    return split_train, split_test


def get_lowest_document_information(root_path=None, targets=None, verbose=True):
    documents_count = []
    for target in targets:
        files = get_all_files(os.path.join(root_path, target))
        if verbose:
            print("Target Directory : {} - Documents : {}".format(target, len(files)))
        documents_count.append(len(files))
    return min(documents_count)
