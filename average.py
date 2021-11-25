from csv import writer

import pandas as pd
import os


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


if __name__ == '__main__':
    results_path = os.path.join('data', 'ann_results_17dim.csv')
    all_results_path = os.path.join('data', 'all_mean_results_17dim.csv')
    result = pd.read_csv(results_path, index_col=[0])
    # print(result.mean())
    model_name = 'ANN final'
    accuracy = round(result.mean()[0], 3)
    print("Accuracy : %f" % accuracy)
    precision = round(result.mean()[1], 3)
    print("Precision Score : %f" % precision)
    recall = round(result.mean()[2], 3)
    print("Recall Score : %f" % recall)
    f1_score = round(result.mean()[3], 3)
    print("F1 Score : %f" % f1_score)
    # exit(0)

    results = [
        0, model_name, accuracy, precision, recall, f1_score]

    append_list_as_row(all_results_path, results)