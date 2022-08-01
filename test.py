import torch
from model.resnet import resnet18_cifar
from model.lenet import *
from model.alexnet import AlexNet
from converter import Converter
from compiler.utils.unique_class_name import refresh_class_name
from tqdm import tqdm
from task.ppu.utils import replace
from compiler.graph.replace_tool import Finder,ReplaceTool

if __name__=="__main__":

    torch_net1 = AlexNet()
    # torch_net1 = resnet18_cifar()
    # torch_net2 = resnet18_cifar()

    in_shape=[4,3,32,32]

    def convert(torch_net):
        converter = Converter(torch_net,in_shape=in_shape)
        converter.convert()
        net = converter.net
        net.reduce_tensor()
        net.deal_input_data()

        replace(net)
    
        replace_tool = ReplaceTool(net=net,config_path="./task/ppu/merge.yaml")
        replace_tool.replace_all()
        return net


    net1 = convert(torch_net1)
    # net2 = convert(torch_net1)
    different = 0
    for i in tqdm(range(200)):
        refresh_class_name()
        net2 = convert(torch_net1)
        if not net1.equals(net2):
            different += 1
        net1 = net2
    print("different:",different)
    # print(net1)

    # refresh_class_name()
    # net2 = convert(torch_net2)

    # print(net1.equals(net2))

