import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from numpy import vstack
from network import ContemptNet
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn

def boolean_df(item_lists, unique_items):
# Create empty dict
    bool_dict = {}

    
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: 1 if item in x else 0)
    result = pd.DataFrame(bool_dict)
    # result['filename'] = filenames
    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

le = LabelEncoder()
def train_model(train_dl, model, criterion, optimizer):
    training_loss = []
    avg_loss = 0
    for i, data in enumerate(train_dl):
        # print()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs, labels = data
        # print(type(labels))
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        model = model.train()
        # print(model)
        # print("input shape: ", inputs.shape)
        # print(input)
        # forward + backward + optimize
        yhat = model(inputs)
        loss = criterion(yhat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())
    return np.mean(training_loss)

# make a class prediction for one row of data

def predict(row, model):
    # convert row to data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    row = torch.tensor([row], dtype=torch.float32).to(device)
    model.eval()
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return np.argmax(yhat)
    

def valid_model(valid_dl, model, criterion):
    valid_loss = []
    valid_acc = []
    predictions, actuals = list(), list()
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(valid_dl):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.float().to(device)
        # retrieve numpy array
        yhat = model(inputs)
        # print("yhat shape: ", yhat.shape)
         # retrieve numpy array
        loss = criterion(yhat, targets)
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        # yhat = np.argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), len(label_cols)))
        # print("Actual: ", actual[0])
        yhat = yhat.reshape((len(yhat),  len(label_cols)))
        yhat = (yhat > 0.5)
        # print("predicted: ", yhat[0])

        # calculate accuracy
        predictions.append(yhat)
        actuals.append(actual)
        acc = metrics.jaccard_score(vstack(predictions), vstack(actuals), average='samples')
        # round to class values
        valid_loss.append(loss.item())
        valid_acc.append(acc)

    # f1_metric = f1_score(vstack(actuals), vstack(predictions), average = "macro")
    # print('F1 score:' , f1_metric)
    jac = metrics.jaccard_score(vstack(actuals), vstack(predictions), average = 'samples')
    print('Jaccard score:' , jac)
    return np.mean(valid_loss), np.mean(valid_acc)

def get_dataloaders(df, batch_size):
        
  
    Y = df[label_cols].values

        
    X = df.drop(metadata_cols + label_cols, axis=1)
    # Y = le.fit_transform(Y)
    # print(le.classes_)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader


batch_size = 32
epochs = 100


data_path = '../new_data/NA/na_dataset.csv'
X_df = pd.read_csv(data_path, index_col=None)

Y_df = pd.read_csv('../new_data/NA/na_labels.csv', usecols=['filename', 'emotions'], index_col='filename')
Y_df["emotions"] = Y_df["emotions"].apply(eval)
unique_items = to_1D(Y_df["emotions"]).unique()
labels_expanded = boolean_df(Y_df['emotions'], unique_items)
# labels_expanded.set_index('filename')
X_df['none']  = np.NaN
X_df['furious']  = np.NaN
X_df['anger']  = np.NaN
X_df['annoyed']  = np.NaN
X_df['contempt']  = np.NaN
X_df['disgust']  = np.NaN
X_df['hatred']  = np.NaN



for index, row in X_df.iterrows():
    # print(index, row)
    filename = X_df.iloc[index]['filename']
    # print(labels_expanded.loc[filename]['none':'hatred'].to_list())
    X_df.at[index,'none'] = labels_expanded.at[filename,'none']
    X_df.at[index,'furious'] = labels_expanded.at[filename,'furious']
    X_df.at[index,'anger'] = labels_expanded.at[filename,'anger']
    X_df.at[index,'annoyed'] = labels_expanded.at[filename,'annoyed']
    X_df.at[index,'contempt'] = labels_expanded.at[filename,'contempt']
    X_df.at[index,'disgust'] = labels_expanded.at[filename,'disgust']
    X_df.at[index,'hatred'] = labels_expanded.at[filename,'hatred']

cols_to_scale = list (
    set(X_df.columns.to_list()) - set(['frame', 'face_id', 'culture', 'filename', 'timestamp', 'confidence','success', 'none', 'furious', 'anger', 'annoyed', 'contempt', 'disgust', 'hatred'])
)
scaler = MinMaxScaler()
X_df[cols_to_scale] = scaler.fit_transform(X_df[cols_to_scale])
metadata_cols = ['frame', 'face_id', 'culture', 'filename', 'timestamp']
label_cols = ['none', 'furious', 'anger', 'annoyed', 'contempt', 'disgust', 'hatred']

test_videos = ['na/vid_6.mp4', 'na/vid_19.mp4', 'na/vid_43.mp4', 'na/vid_25.mp4', 'na/vid_23.mp4', 'na/vid_10_1.mp4', 'na/vid_72.mp4', 'na/vid_34.mp4', 'na/vid_90.mp4', 'na/vid_92.mp4', 'na/vid_39.mp4', 'na/vid_30.mp4', 'na/vid_3.mp4', 'na/vid_33.mp4', 'na/vid_4.mp4', 'na/vid_31.mp4', 'na/vid_53.mp4', 'na/vid_52.mp4', 'na/vid_55.mp4', 'na/vid_59.mp4', 'na/vid_22.mp4', 'na/vid_11.mp4', 'na/vid_79.mp4', 'na/vid_54.mp4', 'na/vid_87.mp4', 'na/vid_63.mp4', 'na/vid_12.mp4', 'na/vid_10_2.mp4', 'na/vid_97.mp4', 'na/vid_70.mp4', 'na/vid_42.mp4', 'na/vid_49.mp4', 'na/vid_77.mp4']
# df = df[df['gender'] == 'male']
kfold = KFold(5, True, 1)
# target_culture = 'Persian'
# df = df[(df['culture'] == target_culture)]
# SPLITTING HELD-OUT DATA
videos = X_df['filename'].unique()


# videos must be array to be subscriptable by a list
videos = np.array(list(set(videos) - set(test_videos)))
# Removing test videos from train dataset
test_df = X_df[X_df['filename'].isin(test_videos)]
y_test = test_df[label_cols].values
X_test = test_df.drop(columns = ['frame', 'face_id', 'culture', 'filename', 'timestamp', 'confidence','success']).values
X_df = X_df[~X_df['filename'].isin(list(test_videos))]
# df = df[df['culture'] != 'Persian']
splits = kfold.split(videos)
kfold_valid_acc = []
kfold_test_acc = []



# TRAINING
test_df_copy = test_df.drop(['frame', 'face_id', 'culture', 'filename', 'confidence','success'], axis=1)
for (i, (train, test)) in enumerate(splits):
    print('%d-th split: train: %d, test: %d' % (i+1, len(videos[train]), len(videos[test])))
    train_df = X_df[X_df['filename'].isin(videos[train])]
    valid_df = X_df[X_df['filename'].isin(videos[test])]

    train_dataloader = get_dataloaders(train_df, batch_size)
    valid_dataloader = get_dataloaders(valid_df, batch_size)
    net = ContemptNet()
    if torch.cuda.is_available():
        net.cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-5)
    optimizer = optim.ASGD(net.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    valid_losses = []
    valid_acc = []
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        epoch_train_loss = train_model(train_dataloader, net, criterion, optimizer)
        epoch_valid_loss, epoch_valid_acc = valid_model(valid_dataloader, net, criterion)
        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)
        print('[%d] Training loss: %.3f' %
                    (epoch + 1, epoch_train_loss ))
        print('[%d] Validation loss: %.3f' %
                    (epoch + 1, epoch_valid_loss ))
        print('[%d] Validation accuracy: %.3f' %
                    (epoch + 1, epoch_valid_acc ))
        valid_acc.append(epoch_valid_acc)
    kfold_valid_acc.append(valid_acc[-1])
    ## Plot the curves
    plt.plot(train_losses, label='training loss')
    plt.plot(valid_losses, label='validation loss')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

    ## Testing
    Yhat = list()
    for row in test_df_copy.values:
        p = predict(row, net)
        Yhat.append(p)

    # test_df['integer_emotion'] = Y
    # test_df['predicted'] = le.inverse_transform(Yhat)
    # print(test_df.sample(n=20))
    # test_acc = accuracy_score(le.fit_transform(test_df['emotion'].values), Yhat)
    test_acc = metrics.jaccard_score(test_df[label_cols].values, Yhat, average='samples')
    # cf_matrix = confusion_matrix(le.fit_transform(test_df['emotion'].values), Yhat)
    # print('********Confusion Matrix*********\n', cf_matrix)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=le.inverse_transform([0,1,2]), columns=le.inverse_transform([0,1,2]))
    # plt.figure(figsize=(10,7))
    # sn.heatmap(df_cm, annot=True, fmt='.2%')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show() 
    # misclassified_test_df = test_df[test_df.predicted != test_df.emotion]
    # test_f1 = f1_score(le.fit_transform(test_df['emotion'].values), Yhat, average='macro')
    kfold_test_acc.append(test_acc)
    print('Test accuracy: %.3f' % (test_acc))
    # print('Test F1-Score: %.3f' % (test_f1))


test_df.to_csv('test_result.csv')
print('Average Test Accuracy on 5-Fold CV: %.3f' % (np.mean(kfold_test_acc)))
print('Average Validation Accuracy on 5-Fold CV: %.3f' % (np.mean(kfold_valid_acc)))