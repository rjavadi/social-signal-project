import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import math
from torch import nn
from torch.nn import functional as F

class LSTM_ContemptNet(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, num_of_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = num_of_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.out_act = nn.Softmax()
        self.batch_size = None
        self.hidden = None

    def forward(self,x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        output_classes = self.out_act(out)
        return output_classes

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]