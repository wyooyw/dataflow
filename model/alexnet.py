import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, n_classes = 10):
        super().__init__()
        # self.conv = nn.Sequential(
        self.conv1 = nn.Conv2d(3, 96, 3, 1, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(96, 256, 3, 1, 1, bias=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool2 = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(256,384, 3, 1, 1, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1, bias=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.Conv2d(384,256, 3, 1, 1, bias=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.maxpool5 = nn.MaxPool2d(2, 2, 0)
        self.flatten = nn.Flatten()
        self.drop6 = nn.Dropout()
        self.line6 = nn.Linear(4096, 2048, bias=False)
        self.relu6 = nn.ReLU(inplace=False)
        self.drop7 = nn.Dropout()
        self.line7 = nn.Linear(2048, 2048, bias=False)
        self.relu7 = nn.ReLU(inplace=False)
        self.line8 = nn.Linear(2048, n_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.flatten(x)
        
        # x = self.drop6(x)
        x = self.line6(x)
        x = self.relu6(x)
        # x = self.drop7(x)
        x = self.line7(x)
        x = self.relu7(x)
        x = self.line8(x)
        return x
if __name__=="__main__":
    print(AlexNet())
    # from torchinfo import summary
    # summary(AlexNet(), input_size=(4, 3, 32, 32))