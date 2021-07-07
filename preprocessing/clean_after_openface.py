import pandas as pd

remove_all = ['vid_10', 'vid_40', 'vid_41', 'vid_71', 'vid_72_1', 'vid_72_2', 'vid_91', 'vid_94', 'vid_96', 'vid_98',
              'vid_99', 'vid_102', 'vid_103', 'vid_105', 'vid_106']

faces_to_keep = {'vid_1':0, 'vid_2':0, 'vid_21':0, 'vid_24':1, 'vid_25':1, 'vid_27':2, 'vid_29':1, 'vid_3':0,
                 'vid_32':0, 'vid_38':0, 'vid_48':2, 'vid_51':0, 'vid_6':1, 'vid_61':1, 'vid_68':0, 'vid_7':0,
                 'vid_70':2, 'vid_72':0, 'vid_73':1, 'vid_77':0, 'vid_80':0, 'vid_81':0, 'vid_83':0, 'vid_84':0,
                 'vid_85':0, 'vid_86':1, 'vid_87':0, 'vid_104':0 }

df = pd.read_csv('../new_data/NA/na_dataset.csv')
# df.drop(df.columns[0], axis=1, inplace=True)
#
for video in remove_all:
    df.drop(df[df.filename == video].index, inplace=True)
#
for k, v in faces_to_keep.items():
    # mask = df['filename'] == k & df['face_id'] == v
    df.drop(df[(df.filename == k) & (df.face_id != v)].index, inplace=True)

df.to_csv('../new_data/na_dataset.csv', index=False)

