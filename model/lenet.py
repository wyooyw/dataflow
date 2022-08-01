import torch.nn as nn
import torch
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1,3,1,stride=1,padding=0,bias=False)
        self.conv2 = nn.Conv2d(1,3,3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(1,1,0)
        self.linear = nn.Linear(108,16,bias=False)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.relu(x)
        x4 = self.pool(x)
        out1 = x1 + x2
        out2 = x3 + x4
        out = out1 + out2
        x = self.flatten(out)
        x = self.linear(x)
        return x
