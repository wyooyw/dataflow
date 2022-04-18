from test_op.communicate.file_communicate import FileCommunicate,FileMessage
import test_op.format.conv as format_conv 
import torch.nn as nn
import numpy as np
class S2TrainInterface(object):
    COMMUNICATE_PATH = "D://communicate/"

    def __init__(self):
        pass

    @staticmethod  
    def execute_conv(self,input,weight,padding=0,stride=1):
        #格式转换
        print("execute conv on S2Train...")
        format_input = format_conv.convert_input(input)
        format_weight = format_conv.convert_weight(weight)

        messages = []
        messages.append(FileMessage("input.txt",format_input))
        messages.append(FileMessage("weight.txt",format_weight))
        attr = {
            "padding":padding,
            "stride":stride
        }
        messages.append(FileMessage("attr.txt",attr))

        file_communicate = FileCommunicate(S2TrainInterface.COMMUNICATE_PATH)
        file_communicate.send(messages) #发送数据给S2Train仿真进程
        result = file_communicate.recv()#从S2Train仿真进程接收数据
        return result

    @staticmethod  
    def execute_linear(self):
        pass