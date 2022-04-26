import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import symbolic_trace,Graph,Node,replace_pattern
import numpy as np

def back_graph_gen(trace):
    pass



backward_ops = {
    nn.Conv2d:BackwardConv2d,
    nn.ReLU:BackwardRelu,
    nn.MaxPool2d:BackwardPool,
    Loss:BackwardLoss
}
class BackwardConv2d(nn.Module):
    def __init__(self):
        super(BackwardConv2d, self ).__init__()

class BackwardRelu(nn.Module):
    def __init__(self):
        super(BackwardRelu, self ).__init__()

class BackwardPool(nn.Module):
    def __init__(self):
        super(BackwardPool, self ).__init__()

class BackwardLoss(nn.Module):
    def __init__(self):
        super(BackwardLoss, self ).__init__()
        
#Test
class Loss(nn.Module):
    def __init__(self):
        super(MyLoss, self ).__init__()
        self.is_leaf = True
    def forward(self,x):
        return x.sum()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self ).__init__() 
        self.conv = nn.Conv2d(1,1,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.loss = Loss()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.loss(x)
        return x

if __name__=="__main__":
    model = MyModel()
    trace = symbolic_trace(model)
    print(trace.graph)
    # back_graph_gen(trace)
    # loss = MyLoss()
    # x = torch.from_numpy(np.arange(0,36).reshape(1,1,6,6))+0.0
    # out = model(x)
    # print(out)