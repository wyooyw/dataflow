import test_op.format.conv as format_conv 
import torch.nn as nn
import torch
import numpy as np
class S2TrainInterface(object):
    """
    假装有个仿真进程存在，实际调用pytorch的算子来计算
    用于自己测试
    等S2Train仿真那边做出来后，换成用S2TrainInterface.py里的那个类
    """
    COMMUNICATE_PATH = "D://communicate/"

    def __init__(self):
        pass

    @staticmethod  
    def execute_conv(input,weight,**kwargs):
        print("execute conv2d on S2Train...")
        input = torch.from_numpy(input)
        weight = torch.from_numpy(weight)
        conv = nn.Conv2d(**kwargs)
        conv.weight = nn.Parameter(weight)
        output = conv(input)
        output = output.detach().numpy()
        return output

    @staticmethod  
    def execute_linear():
        pass