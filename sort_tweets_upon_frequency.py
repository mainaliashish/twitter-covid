"""
-- author: Ashish Mainali
-- email: mainaliashish@outlook.com
-- date: 2021-07-14
"""

import pandas as pd

print('Loading Fasttext model..\nPlease wait..')

# Actual csv data
# df = pd.read_csv('datasets.csv')
# Test csv data with less columns
df = pd.read_csv('testdata.csv')
# Load words based on frequencies
frequency = pd.read_csv("words_frequency.csv")


def convert_sentences_to_list(string):
    li = list(string.split(","))
    return li


mergedlist = []
words_freq = dict({
    'words': [],
    'counts': []
})

data_df = dict({
    'initial_tweets': [],
    'final_tweets': [],
})

final_tweet = []

i = 0
tweets = df['Tokanize_tweet']

if i < df.shape[0]:
    for index, item in enumerate(tweets):
        tokens = convert_sentences_to_list(item)
        for token in tokens:
            for ind, row in frequency.iterrows():
                if token == row[1]:
                    print(f'Token matched {token} --> {row[2]}')
                    words_freq['words'].append(row[1])
                    words_freq['counts'].append(row[2])
        print("*************************************")
        print("Sorting current column based on their frequency..✅")

        words_freq = pd.DataFrame(words_freq)
        words_freq = words_freq.sort_values(by="counts", ascending=False)

        final_str = ','.join(words_freq['words'])



        data_df['initial_tweets'].append(item)
        data_df['final_tweets'].append(final_str)

        words_freq = dict({
            'words': [],
            'counts': []
        })
        print("Sorted Successfully..\nAdded sorted column to dictionary..✅")
        print("Operating into next column..👋")

    my_df = pd.DataFrame(data_df)
    my_df.to_csv('new.csv')

print("Operation completed Successfully...✅")
print("Sorted all tokens based on their frequencies.✅")
print("Results have been saved to a csv file.")
exit(0)
