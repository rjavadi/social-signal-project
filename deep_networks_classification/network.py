import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContemptNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=24, out_features=32)
        self.dout = nn.Dropout(0.2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=32, out_features=16)
        # self.bn2 = nn.BatchNorm1d(num_features=32)
        self.dout2 = nn.Dropout(0.2)
        self.prelu2 = nn.PReLU() # parametric relu - if x < 0, returns 0.25*x - o.w. return x

        self.fc3 = nn.Linear(in_features=16, out_features=12)
        self.bn3 = nn.BatchNorm1d(num_features=12)
        # self.dout3 = nn.Dropout(0.5)
        self.prelu3 = nn.PReLU()

        # self.fc4 = nn.Linear(in_features=160, out_features=60)
        # # self.bn4 = nn.BatchNorm1d(num_features=60)
        # self.dout4 = nn.Dropout(0.2)
        # self.prelu4 = nn.PReLU()

        self.out = nn.Linear(in_features=12, out_features=7)
        self.out_act = nn.Sigmoid()

        # self.out_act = nn.Softmax()

    def forward(self, input):
        input = input.view(input.size(0), -1)
        layer1 = self.fc1(input)
        layer1_act = self.relu1(layer1)

        layer2 = self.fc2(layer1)
        layer2_act = self.prelu2(self.dout2(layer2))
        # layer2_dout = self.dout(layer2_act)

        # # layer2_out = self.out(layer2_act)

        layer3 = self.fc3(layer2_act)
        layer3_act = self.prelu3(self.bn3(layer3))
        # layer3_out = self.out(layer3_act)

        # layer4 = self.fc4(layer3_out)
        # layer4_act = self.prelu3(self.dout4(layer4))
        layer4_out = self.out(layer3_act)

        output_classes = self.out_act(layer4_out)
        return output_classes
        
