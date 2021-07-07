import pandas as pd
import numpy as np
import os
import re
from glob import glob

"""
"""
def concat_files(records: str):

    head_r_cols = ["pose_Rx", "pose_Ry", "pose_Rz"]
    gaze_dir_cols = ["gaze_angle_x", "gaze_angle_y"]
    au_cols = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
               "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
    basic_cols = ["filename", "culture", "frame", "face_id", "timestamp","confidence", "success" ]

    to_keep = basic_cols  + au_cols + head_r_cols + gaze_dir_cols

    records = sorted(records)
    res_df = None
    for i in range(len(records)):
        tmp = pd.read_csv(records[i])
        # print(records[i], len(tmp.index))
        print('************************')
        file_name = os.path.splitext(os.path.basename(records[i]))[0]
        tmp['filename'] = file_name
        print('file name:', file_name)
        tmp['culture'] = 'north american'
        if file_name.find('?') < 0:
            # tmp['emotion'] = file_name[:file_name.find('_')]
            res_df = pd.concat([res_df, tmp], ignore_index=True, sort=False)

        # print(train.head())
        
        
        # au_columns = res_df.columns[df.columns.str.contains(au_regex_pat)]

    # au_regex_pat = re.compile(r'^cdAU[0-9]+_r$')     
    res_df.rename(columns=lambda x: x.strip(), inplace=True)
    # emma_df = pd.read_csv('../all_videos.csv')
    res_df = res_df.filter(to_keep)
    res_df.drop(res_df[res_df['confidence'] < 0.85].index, inplace=True)
    res_df.drop(res_df[res_df['success'] == 0].index, inplace=True)
    res_df.to_csv(os.path.join(os.path.curdir, '../new_data/NA/na_dataset.csv'), index=False)

records = glob('/home/roya/Project/Processed_videos_na/*.csv')
print('**************   Records: ', records)
concat_files(records)


###### Adding label column

# records = glob("./disgust_processed_csv_p/*.csv")
# res_df = pd.read_csv("./processed_data_csv/all_videos.csv")
# for i in range(len(records)):
#     tmp = pd.read_csv(records[i], index_col=[0])
#     tmp.rename(columns=lambda x: x.strip(), inplace=True)
#     tmp = tmp.filter(res_df.columns)
#     # print(records[i], len(tmp.index))
#     print('************************')
#     file_name = os.path.splitext(os.path.basename(records[i]))[0]
#     if file_name.find('anger') >= 0:
#         tmp['emotion'] = 'anger'
#     elif file_name.find('disgust') >= 0:
#         tmp['emotion'] = 'disgust'
#     elif file_name.find('contempt') >= 0:
#         tmp['emotion'] = 'contempt'
    
#     res_df = pd.concat([res_df, tmp], sort=False)
# res_df.to_csv('./processed_data_csv/all_videos.csv', index=False)

# https://drive.google.com/file/d/1jhsP2PkX4dMpB9LbXJVawvn1ZaSph7MF/view?usp=sharing

# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jhsP2PkX4dMpB9LbXJVawvn1ZaSph7MF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jhsP2PkX4dMpB9LbXJVawvn1ZaSph7MF" -O kevin_grabage.zip && rm -rf /tmp/cookies.txt