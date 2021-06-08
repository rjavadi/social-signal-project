

import pandas as pd

emotion_data = pd.read_csv("./new_data/labels.csv")
# cols = ["id", "filename","emotions","emoji","gender","confidence","comment","intensity"]
contents = []
emotion_data['emotions'] = emotion_data['emotions'].apply(eval)
for i in range(0, emotion_data.shape[0]):
    row = emotion_data.iloc[i]
    # print(type(row['emotions']))
    for emotion in row['emotions']:
        # for emoji in row['emoji'].split(','):
        contents.append([row["filename"], emotion])
        # print(emotion)
flattened_df = pd.DataFrame(columns=['filename', 'emotion'], data=contents)

emoji_data = pd.read_csv("./new_data/emoji_labels.csv", index_col='filename')
emoji_data['emoji'] = emoji_data['emoji'].apply(eval)
cols = flattened_df['emotion'].unique()
rows = ['angry', 'furious', 'hatred', 'neutral', 'rollingeyes', 'smirk', 'unamused', 'weary', 'smilingimp', 
    'vomiting', 'triumph', 'none', 'expressionless', 'nauseated', 'skeptical']

df = pd.DataFrame(index=rows, columns=cols)
df.fillna(0, inplace=True)
for i in range(0, flattened_df.shape[0]):
    row = flattened_df.iloc[i]
    # print(row)
    for emoji in emoji_data.loc[row['filename']]['emoji']:
        df.loc[emoji, row['emotion']] += 1


# Normalizing rows:
df["sum"] = df.sum(axis=1)
df_new = df.loc[:,"annoyed":"furious"].div(df["sum"], axis=0)
df_new

# Plotting heatmap
import seaborn as sns
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize = (9,5))
sns.heatmap(df.loc[:, "annoyed":"furious"], annot=True)
plt.savefig("heatmap.png", dpi = 300)