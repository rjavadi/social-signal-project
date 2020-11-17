import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from numpy import vstack
from lstm_network import LSTM_ContemptNet
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sn
# from torch.nn import functional as F
from dataloader import create_timeseries, create_dataloader

def train_model(train_dl, model, criterion, optimizer):
    training_loss = []
    avg_loss = 0
    for i, (x_batch, y_batch) in enumerate(train_dl):
        # print()

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        model = model.train()
        # forward + backward + optimize
        yhat = model(x_batch)
        loss = criterion(yhat, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())
    return np.mean(training_loss)

def valid_model(valid_dl, model, criterion):
    valid_loss = []
    valid_acc = []
    predictions, actuals = list(), list()
    model.eval()
    for _, (x_val, y_val) in enumerate(valid_dl):
        x_val = x_val.cuda()
        y_val = y_val.cuda()
        # retrieve numpy array
        yhat = model(x_val)
        # print("yhat shape: ", yhat.shape)
         # retrieve numpy array
        loss = criterion(yhat, y_val)
        yhat = yhat.cpu().detach().numpy()
        actual = y_val.cpu().numpy()
        # convert to class labels
        yhat = np.argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # calculate accuracy
        predictions.append(yhat)
        actuals.append(actual)
        acc = accuracy_score(vstack(predictions), vstack(actuals))
        # round to class values
        valid_loss.append(loss.item())
        valid_acc.append(acc)

    f1_metric = f1_score(vstack(actuals), vstack(predictions), average = "macro")
    print('F1 score:' , f1_metric)
    return np.mean(valid_loss), np.mean(valid_acc)

def predict(row, model):
# convert row to data
    row = torch.tensor([row], dtype=torch.float32).cuda()
    model.eval()
    # make prediction after adding batch dimenstion
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return np.argmax(yhat)

batch_size = 32
epochs = 100
input_dim = 16    
hidden_dim = 64
num_of_layers = 3
output_dim = 9
seq_dim = 50

learning_rate = 0.0005

df = pd.read_csv('../videos_relabelled.csv', index_col=None)
videos = df['filename'].unique()
test_videos = pd.Series(videos).sample(frac=0.20)

# videos must be array to be subscriptable by a list
videos = np.array(list(set(videos) - set(test_videos)))
kfold = KFold(5, True, 1)
splits = kfold.split(videos)
kfold_valid_acc = []
kfold_test_acc = []

# Removing test videos from train dataset
test_df = df[df['filename'].isin(test_videos)]
for (i, (train, test)) in enumerate(splits):
    print('%d-th split: train: %d, test: %d' % (i+1, len(videos[train]), len(videos[test])))
    train_df = df[df['filename'].isin(videos[train])]
    valid_df = df[df['filename'].isin(videos[test])]

    X_train, y_train = create_timeseries(train_df)
    X_test, y_test = create_timeseries(test_df)
    train_dataloader = create_dataloader(X_train, y_train, batch_size)
    valid_dataloader = create_dataloader(X_test, y_test, batch_size)
    net = LSTM_ContemptNet()
    print(net)
    if torch.cuda.is_available():
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    # TODO: if result are not good, change to RMSprop
    optimizer = optim.ASGD(net.parameters(), lr=learning_rate)

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
    X_test_list, Y_test_list = create_timeseries(test_df)
    test_dl = create_dataloader(X_test_list, Y_test_list, batch_size=1)
    for batch,_ in test_dl:
        # batch = batch.permute(0, 2, 1)
        y_hat = net(batch.cuda())
        Yhat += y_hat.tolist()

    le = LabelEncoder()
    Yhat = le.inverse_transform(Yhat)
    print(test_df.sample(n=20))
    test_acc = accuracy_score(le.fit_transform(Y_test_list), Yhat)
    cf_matrix = confusion_matrix(le.fit_transform(Y_test_list), Yhat)
    print('********Confusion Matrix*********\n', cf_matrix)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=le.inverse_transform([0,1,2]), columns=le.inverse_transform([0,1,2]))
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, fmt='.2%')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show() 

    # misclassified_test_df = test_df[test_df.predicted != test_df.emotion]
    test_f1 = f1_score(le.fit_transform(Y_test_list), Yhat, average='macro')
    kfold_test_acc.append(test_acc)
    print('Test accuracy: %.3f' % (test_acc))
    print('Test F1-Score: %.3f' % (test_f1))


# test_df.to_csv('test_result.csv')
print('Average Test Accuracy on 5-Fold CV: %.3f' % (np.mean(kfold_test_acc)))
print('Average Validation Accuracy on 5-Fold CV: %.3f' % (np.mean(kfold_valid_acc)))
