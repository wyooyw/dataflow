import torch.nn as nn
import dataflow.function.conv.S2TrainConv2dFunction as S2TrainConv2dFunction
"""
卷积算子
真正的计算会发送到S2Train仿真进程上完成
"""
class S2TrainConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        self.weight = nn.Parameter(nn.rand((out_channels,in_channels,kernel_size,kernel_size)))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
    def forward(self,x):
        return S2TrainConv2dFunction.apply(x)

