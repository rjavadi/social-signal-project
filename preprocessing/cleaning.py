import pandas as pd

data = pd.read_csv("../new_data/annotation.csv")

# sorting by first name
# data.sort_values("id", inplace = True)

# dropping ALL duplicte values
data.drop_duplicates(subset =["filename","emotions","gender","confidence","comment","emoji","annotator_individuality","intensity"],
                     keep = 'first', inplace = True)
data.to_csv("../new_data/annotations_wo_dupes.csv", index=False)

# Separate NA and Persian annotations
# Also removes video culture column
data.drop(columns=['video_culture'], inplace=True)
p_mask = data['annotator_culture'] == 'persian'
na_mask = data['annotator_culture'] == 'north american'

persian_data = data[p_mask]
na_data = data[na_mask]
persian_data.to_csv("../new_data/persian_annotations.csv", index=False)
na_data.to_csv("../new_data/na_annotations.csv", index=False)