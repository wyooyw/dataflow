import torch.nn as nn
import torch

class Nop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes,affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3( planes, planes)
        self.bn2 = nn.BatchNorm2d(planes,affine=True)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes,affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes,affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,affine=True)
        self.relu3 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class ResNet_cifar(nn.Module):

    def __init__(self, block, layers,num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.block_index = 0
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine=True)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Nop(),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                Nop(),
                nn.BatchNorm2d(planes * block.expansion,affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes , stride, downsample))
        self.inplanes = planes * block.expansion
        block_index = 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            block_index += 1 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("relu.output:",x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print("maxpool.input:",x.shape)
        x = self.avgpool(x)
        x = self.flatten(x)
        # print("fc.input:",x.shape)
        x = self.fc(x)

        return x


def resnet18_cifar(num_classes = 10):

    model = ResNet_cifar(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)
    return model

if __name__=="__main__":
    print(resnet18_cifar())