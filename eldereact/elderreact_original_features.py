import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.validation import _check_large_sparse
from sklearn.metrics import pairwise_distances_argmin_min, jaccard_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


def get_features(filename):
    features = []
    video_feat_file = open(filename).readlines()
    # print(len(video_feat_file[5]))
    for i in range(len(video_feat_file)):
        video_feat = video_feat_file[i].strip().split("\t")
        
        video_feat = video_feat[1].split()
        # print(video_feat)
        v = [float(i) for i in video_feat]
        # video_feat = v[0:8]+v[288:294]+v[634:717]+v[997:1003]+v[1343:]
        video_feat = v[288:294]+v[997:1003]+v[1343:]

        features.append(np.array(video_feat))
    # print(len(video_feat))
    return features


def create_label_dict(filename:str):
    label_file = open(filename).readlines()
    labels = []
    filenames = []
    for i in range(len(label_file)):
        filenames.append(label_file[i].split(" ")[0])
        labels.append([int(l) for l in label_file[i].split(" ")[1:7]])
        # labels.append([random.randint(0, 1) for i in range(6)])
    return labels

test_feats = get_features("./test_video.txt")
train_feats = get_features("./train_video.txt")
train_labels = create_label_dict("./train_labels.txt")
test_labels = create_label_dict("./test_labels.txt")

train_feats = np.array(train_feats)
test_feats = np.array(test_feats)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(len(train_feats))
print(train_feats[0].shape)
affects = ['anger','disgust','fear','happy','sad','surprise']

kfold = KFold(5, True, 1)
splits = kfold.split(train_feats)
best_model_score = 0
best_model = None
for (i, (train, valid)) in enumerate(splits):
    print(train.dtype)
    X_train = train_feats[train.astype(int)]
    y_train = train_labels[train.astype(int)]
    # print('y train: ', X_train)

    X_valid = train_feats[valid.astype(int)]
    y_valid = train_labels[valid.astype(int)]

    base_xgb = XGBClassifier(objective="binary:logistic", eval_metric='logloss')
    chains = [ClassifierChain(base_xgb, order='random', random_state=i)
            for i in range(5)]
    best_model_index = 0
    best_jac = 0            
    for j, model in enumerate(chains):
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        val_score =jaccard_score(y_valid, valid_pred, average='samples')
        if val_score > best_jac:
            best_model_index = j
            best_jac = val_score
            

    valid_pred_chains = chains[best_model_index].predict(X_valid)
    chain_jaccard_scores = jaccard_score(y_valid, valid_pred_chains >= .5, average='samples')
    if chain_jaccard_scores > best_model_score:
        best_model_score = chain_jaccard_scores
        best_model = chains[best_model_index]

Y_pred = best_model.predict(test_feats)
y_test = test_labels
chain_jaccard_scores = jaccard_score(y_test, Y_pred >= .5, average='samples')
print(metrics.classification_report(y_test,  Y_pred, target_names=affects))

                    
    
    
    # print("CC Validation Jaccard Score:\n ", chain_jaccard_scores)
