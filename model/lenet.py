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
        # self.conv = nn.Conv2d(2,3,2,stride=1,padding=0,bias=False)
        # self.conv.weight = nn.Parameter(torch.range(1.0,12.0).reshape(3,1,2,2))
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(3,4,2,stride=1,padding=1,bias=False)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(4,5,2,stride=2,padding=0,bias=False)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(5,6,2,stride=2,padding=1,bias=False)
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
        pass
    def hooker(self,grad):
        print(grad)

    def forward(self,x):
        # x = self.conv(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        x = self.bn(x)
        x = self.flatten(x)
        # x = self.linear(x)
        # x = self.relu(x)
        # x = self.linear2(x)
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