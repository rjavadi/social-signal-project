import pandas as pd


# def clean(input: str):
#     input = input.replace(',', '","')
#     input = '["' + input
#     input = input+'"]'
#     return input

# df = pd.read_csv('../new_data/Persian CAD social signals.csv')
# df['social_signals'] = df['social_signals'].apply(clean)
# df['social_signals'] = df['social_signals'].apply(eval)
# df.to_csv('../new_data/ss_persian.csv', columns=['filename', 'social_signals'], index=False)



emoji_data = pd.read_csv("../new_data/emoji_labels.csv")
contents = []
emoji_data['emoji'] = emoji_data['emoji'].apply(eval)
for i in range(0, emoji_data.shape[0]):
    row = emoji_data.iloc[i]
    # print(type(row['emotions']))
    # for emotion in row['emotions']:
    for emoji in row['emoji']:
        contents.append([row["filename"], emoji])
        # print(emotion)
flattened_df = pd.DataFrame(columns=['filename', 'emoji'], data=contents)

ss_data = pd.read_csv("../new_data/ss_persian.csv", index_col='filename')
ss_data['social_signals'] = ss_data['social_signals'].apply(eval)
cols = flattened_df['emoji'].unique()
rows = ['Wrinkled nose', 'Eyebrows Pushed Together', 'Side Eye', 'Curling Upper Lip', 'Raised Eyebrows', 'Lips Pressed Together', 'Mocking', 'Shaking Head', 'Smirk', 
    'Head Turned Away', 'Calm', 'Wide Eyes', 'Snarl', 'Raised chin', 'Squinting', 'Smiling', 'Closing Eyes', 'Arms Crossed']

df = pd.DataFrame(index=rows, columns=cols)
df.fillna(0, inplace=True)
for i in range(0, flattened_df.shape[0]):
    row = flattened_df.iloc[i]
    # print(row)
    for signal in ss_data.loc[row['filename']]['social_signals']:
        df.loc[signal, row['emoji']] += 1


# Normalizing rows:
df["sum"] = df.sum(axis=1)
df_new = df.loc[:,cols[0]:cols[-1]].div(df["sum"], axis=0)
df_new

# Plotting heatmap
import seaborn as sns
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize = (9,5))
sns.heatmap(df.loc[:, cols[0]:cols[-1]], annot=True)
plt.savefig("ss_emoji_persian_heatmapt.png", dpi = 300)