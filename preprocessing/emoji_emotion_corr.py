

import pandas as pd

# data = pd.read_csv("../new_data/na_labels.csv")
# # cols = ["id", "filename","emotions","emoji","gender","confidence","comment","intensity"]
# contents = []
# data['emotions'] = data['emotions'].apply(eval)
# for i in range(0, data.shape[0]):
#     row = data.iloc[i]
#     # print(type(row['emotions']))
#     for emotion in row['emotions']:
#         # for emoji in row['emoji'].split(','):
#         contents.append([row["filename"], emotion])
#         # print(emotion)
# flattened_df = pd.DataFrame(columns=['filename', 'emotion'], data=contents)

# emoji_data = pd.read_csv("../new_data/na_emoji_labels.csv", index_col='filename')
# emoji_data['emoji'] = emoji_data['emoji'].apply(eval)
# cols = ['annoyed','contempt','anger','none','hatred','disgust','furious']
# rows = ['angry', 'furious', 'hatred', 'neutral', 'rollingeyes', 'smirk', 'unamused', 'weary', 'smilingimp', 
#     'vomiting', 'triumph', 'none', 'expressionless', 'nauseated', 'skeptical']

# df = pd.DataFrame(index=rows, columns=cols)
# df.fillna(0, inplace=True)
# for i in range(0, flattened_df.shape[0]):
#     row = flattened_df.iloc[i]
#     # print(row)
#     for emoji in emoji_data.loc[row['filename']]['emoji']:
#         df.loc[emoji, row['emotion']] += 1


# # Normalizing rows:
# # df["sum"] = df.sum(axis=1)
# # df_new = df.loc[:,"annoyed":"furious"].div(df["sum"], axis=0)
# # df_new

# # Plotting heatmap
# import seaborn as sns
# import matplotlib.pyplot as plt 

# fig, ax = plt.subplots(figsize = (9,5))
# sns.heatmap(df.loc[:, "annoyed":"furious"], annot=True)
# plt.savefig("na_after_voting_heatmap.png", dpi = 300)





###############    Emotion-Emoji Co-occurrence before voting. ###############

data = pd.read_csv("../new_data/NA/na_annotations.csv")
# cols = ["id", "filename","emotions","emoji","gender","confidence","comment","intensity"]

def clean(input: str):
    input = input.replace(',', '","')
    input = '["' + input
    input = input+'"]'
    return input


data['emotions'] = data['emotions'].apply(clean)
data['emoji'] = data['emoji'].apply(clean)
# df.to_csv('../new_data/ss_persian.csv', columns=['filename', 'social_signals'], index=False)


data['emotions'] = data['emotions'].apply(eval)
data['emoji'] = data['emoji'].apply(eval)

cols = ['anger', 'annoyed', 'contempt', 'disgust', 'hatred', 'furious', 'none']
rows = ['angry', 'furious', 'hatred', 'neutral', 'rollingeyes', 'smirk', 'unamused', 'weary', 'smilingimp', 
    'vomiting', 'triumph', 'none', 'expressionless', 'nauseated', 'skeptical']

count_df = pd.DataFrame(index=rows, columns=cols)

count_df.fillna(0, inplace=True)

for i in range(0, data.shape[0]):
    row = data.iloc[i]
    # print(type(row['emotions']))
    for emotion in row['emotions']:
        for emoji in row['emoji']:
            count_df.loc[emoji, row['emotions']] += 1
           

# Normalizing rows:
count_df["sum"] = count_df.sum(axis=1)
df_new = count_df.loc[:,"anger":"none"].div(count_df["sum"], axis=0)

# Plotting heatmap
import seaborn as sns
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize = (9,5))
sns.heatmap(count_df.loc[:, "anger":"none"], annot=True)
plt.savefig("na_before_voting_heatmap.png", dpi = 300)