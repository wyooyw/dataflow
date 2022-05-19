import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,6,5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1,10)
        pass
    def forward(self,x):
        # x = self.conv(x)
        # x = self.relu(x)
        # x = self.pool(x)
        # x = self.flatten(x)
        # x = self.linear(x)
        # x = x.relu()
        # x = self.relu(x)
        # x = torch.relu(x)
        x = x+x
        x = torch.add(x,x)
        x = x.add(x)
        return x

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,5)

        self.conv2 = nn.Conv2d(3,3,5)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(3,3,5)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        pass
    def forward(self,x):
        x = self.conv1(x)
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu2(x)
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.relu3(x)
        x = self.flatten(x)
        return x

class TestNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,5)
        self.bn1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(3,3,5)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        pass
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu2(x)
        x = self.flatten(x)
        return x


class TestNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,5)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(3,3,5)
        self.conv3 = nn.Conv2d(3,3,5)

        self.flatten = nn.Flatten()
        pass
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        y = self.bn1(x)
        y = self.conv2(y)
        z = self.bn2(x)
        z = self.conv3(z)
        x = y+z
        x = self.flatten(x)
        return x


class TestNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        pass
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        return x

class TestNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(3,3,3,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(3,affine=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3,3,3,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(3,affine=True)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(588,10,bias=False)
        pass
    def forward(self,x):
        # x = self.relu0(x)
        # identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        # x = x + identity
        x = self.relu2(x)
        x = self.flatten(x)
        # x = self.linear(x)
        return x