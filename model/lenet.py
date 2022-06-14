import torch.nn as nn
import torch
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,5,stride=1,padding=0,bias=False)
        # self.conv.weight = nn.Parameter(torch.range(1.0,12.0).reshape(3,1,2,2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3,3,3,stride=1,padding=1,bias=False)
        self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(3,3,5,stride=2,padding=0,bias=False)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(3,3,5,stride=2,padding=0,bias=False)
        # self.relu4 = nn.ReLU()
        self.bn = nn.BatchNorm2d(3,momentum=0.0)
        self.bn.eval()
        # self.bn.running_mean = torch.randn(3)
        # self.bn.running_var = torch.range(1.0,3.0)+torch.randn(3)
        # self.bn.weight = nn.Parameter(torch.randn(3))
        # self.bn.bias = nn.Parameter(torch.randn(3))

        # self.linear = nn.Linear(108,16,bias=False)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(16,10,bias=False)
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        identity = x
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + identity
        # y = x
        # x = self.conv3(x)
        # x = self.relu3(x)

        # y = self.conv4(y)
        # y = self.relu4(y)

        # x = x + y
        x = self.bn(x)
        x = self.flatten(x)
        # x = self.linear(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        return x
