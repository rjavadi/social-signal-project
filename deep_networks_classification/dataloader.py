import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

enc = LabelEncoder()


def create_timeseries(df: DataFrame):
    grouped = df.groupby(by=['filename', 'face_id'])
    # X_list is video/face frames, divided into 50 frames chunks
    X_list = []
    Y_list = []
    frame_limit = 50
    for key in grouped.groups:
        X_group = grouped.get_group(key)
        if (len(X_group) >= frame_limit):
            for i in range(0, math.floor(len(X_group)/25)):
                splitted_group = X_group[i * 25: min(len(X_group), i * 25 + 50)]
                X_list.append(splitted_group.drop(columns=['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1).values)
                Y_list.append(X_group['emotion'].iloc[0])
        else:
            # TODO: separate video file name, frame and face and store it in a list (video_info_list)
            X_list.append(X_group.drop(columns=['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1).values)
            Y_list.append(X_group['emotion'].iloc[0]) 
    return X_list, Y_list

# After splitting the test data and train-validation data, we create dataset
def create_dataloader(X_list, y_list, batch_size):
    y_list = enc.fit_transform(y_list)
    X_tensor = [torch.tensor(x, dtype=torch.float32) for x in X_list]
    Y_tensor = [torch.tensor(y, dtype=torch.long) for y in y_list]
    # TODO: fix the bug: https://discuss.pytorch.org/t/how-to-turn-list-of-varying-length-tensor-into-a-tensor/1361/2
    ds = TensorDataset(X_tensor, Y_tensor)

    return DataLoader(ds, batch_size=batch_size, shuffle=True)
    
def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()





