"""
-- author: Ashish Mainali
-- email: mainaliashish@outlook.com
-- date: 2021-07-22
"""

import pandas as pd
import os


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


def pad(tokens, content, width):
    tokens.extend([content] * (width - len(tokens)))
    return tokens


def Average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    read_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'test.csv')
    # read_path = os.path.join('data', 'testdata.csv')
    write_path = os.path.join('data', 'datazip', 'fold_dataset', 'set_10', 'test_pad.csv')
    data = pd.read_csv(read_path)
    tokens = data['Tokenize_tweet']
    new_data = dict({
        'Label': [],
        'old_tokens': [],
        'padding_tokens': [],
    })

    # a_list = []

    for index, row in enumerate(tokens):
        tokens_list = convert_sentences_to_list(row)
        if len(tokens_list) > 2:
            # print(type(tokens_list))
            # print(len(tokens_list))
            # a_list.append(len(tokens_list))
            new_list = pad(tokens_list[0:12], 0, 12)
            print(new_list)
            # exit(0)
            new_data['Label'].append(data['Label'][index])
            new_data['old_tokens'].append(','.join([str(elem) for elem in tokens_list]))
            new_data['padding_tokens'].append(','.join([str(elem) for elem in new_list]))

    df_tweets = pd.DataFrame(new_data)
    df_tweets.to_csv(write_path)

# print("Average is %f" % Average(a_list))
print("Tokens have been successfully processed..")
