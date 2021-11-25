import pandas as pd

df = pd.read_csv('new.csv')
data_df = dict({
    'initial_tokens': [],
    'final_tokens': [],
})

alltokens = []
a = 0
b = 1
limit = 0

for index, final_tweet in enumerate(df['final_tweets']):
    tokens = final_tweet.split(',')
    if limit <= 3:
        for _index, token in enumerate(tokens):
            alltokens.append(token)
        data_df['initial_tokens'].append(final_tweet)
        data_df['final_tokens'].append(tokens[0:4])
        limit += 1
print(len(set(alltokens)))
    # else:
    #     limit = 0
    #     # alltokens.clear()

my_df = pd.DataFrame(data_df)
my_df.to_csv("final_tokens_popular.csv")
